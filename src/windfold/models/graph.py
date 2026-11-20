"""graph networks after Daenens et al. 2025: GENConv with softmax aggregation on dense tensors.

each sample is one timestamp: a node per turbine of the farm, padded, with a mask of the valid
ones. edges join valid turbines within edge_distance_d rotor diameters. absolute edges carry the
distance and the sine and cosine of the bearing from receiver to source, and nodes the reference
speed and the sine and cosine of the farm direction. wind-frame edges carry along and across
distances and nodes the reference speed only, so rotating layout and wind together changes
nothing. with measured inflow every node also carries the log of its timestamp's turbulence ratio
(windfold.inflow). outputs are centered over each sample's valid nodes inside the loss and in
predictions.
"""

import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
from torch import nn

from windfold import inflow
from windfold.config import FarmConfig, ModelSpec, Protocol, TurbineConfig
from windfold.geometry import bearings, distances, relations
from windfold.models.farm import Farm, read_farm, timestamp_groups
from windfold.models.interface import FitContext, check_whole_timestamps

Edges = Literal["absolute", "wind_frame"]
EPSILON = 1e-7
PREDICT_BATCH = 1024


@dataclass(frozen=True)
class GraphSettings:
    """one menu configuration of the graph networks."""

    width: int
    layers: int
    gnn_dropout: float
    encoder_dropout: float
    rounds: int
    batch_size: int
    learning_rate: float
    learning_rate_decay: float
    weight_decay: float

    @classmethod
    def from_menu(cls, configuration: Mapping[str, Any]) -> "GraphSettings":
        return cls(**{field.name: configuration[field.name] for field in fields(cls)})


def encoder(inputs: int, width: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width)
    )


class GenConv(nn.Module):
    """h_i <- MLP(h_i + sum_j softmax_j(m_ij) m_ij), m_ij = ReLU(h_j + e_ij) + eps (Li et al. 2020).

    the softmax runs over each receiver's valid neighbors, separately per channel; a node with
    none receives zero. the MLP has two layers, the hidden one twice as wide and layer-normalized.
    """

    def __init__(self, width: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.LayerNorm(2 * width),
            nn.ReLU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, h: torch.Tensor, e: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.relu(h[:, None, :, :] + e) + EPSILON
        mask = adjacency[..., None]
        weights = torch.softmax(message.masked_fill(~mask, -1e9), dim=2) * mask
        return self.mlp(h + (weights * message).sum(dim=2))


class Network(nn.Module):
    def __init__(self, node_inputs: int, edge_inputs: int, settings: GraphSettings):
        super().__init__()
        self.nodes = encoder(node_inputs, settings.width, settings.encoder_dropout)
        self.edges = encoder(edge_inputs, settings.width, settings.encoder_dropout)
        self.convs = nn.ModuleList(GenConv(settings.width) for _ in range(settings.layers))
        self.dropout = nn.Dropout(settings.gnn_dropout)
        self.head = nn.Linear(settings.width, 1)

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        h = self.nodes(nodes)
        e = self.edges(edges).expand(adjacency.shape[0], -1, -1, -1)
        for k, conv in enumerate(self.convs):
            h = conv(h, e, adjacency)
            if k < len(self.convs) - 1:
                h = self.dropout(torch.relu(h))
        return self.head(h).squeeze(-1)


def centred_loss(f: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """mean over valid nodes of (f - mean f - y)^2, the mean taken over each sample's valid
    nodes; y is centered the same way, so a common shift of a sample's f changes nothing."""
    weight = valid.to(f.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
    level = (f * weight).sum(dim=1, keepdim=True) / count
    return (((f - level - y) * weight) ** 2).sum() / weight.sum().clamp(min=1.0)


def centred_output(f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(f.dtype)
    level = (f * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp(min=1.0)
    return f - level


@dataclass
class Samples:
    """S samples over the farm's n turbines: node inputs per sample (the same for every node),
    a table of edge features each sample points into, the pairs close enough to be edges and
    each node's validity and target. adjacency is formed per batch, never stored."""

    inputs: torch.Tensor
    edges: torch.Tensor
    edge_of: torch.Tensor
    near: torch.Tensor
    valid: torch.Tensor
    target: torch.Tensor

    def to(self, device: torch.device) -> "Samples":
        return Samples(*(getattr(self, field.name).to(device) for field in fields(self)))

    def take(self, index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        valid = self.valid[index]
        count = valid.shape[1]
        nodes = self.inputs[index][:, None, :].expand(-1, count, -1)
        edges = self.edges if self.edges.shape[0] == 1 else self.edges[self.edge_of[index]]
        adjacency = self.near[None, :, :] & valid[:, :, None] & valid[:, None, :]
        return nodes, edges, adjacency, valid, self.target[index]


@dataclass(frozen=True)
class GraphInputs:
    """how raw quantities become network inputs: fixed scales, never fitted to data."""

    edges: Edges
    edge_distance_d: float
    speed_scale_ms: float
    measured_inflow: bool = False

    @property
    def node_inputs(self) -> int:
        return (3 if self.edges == "absolute" else 1) + int(self.measured_inflow)


def samples(
    farm: Farm,
    inputs: GraphInputs,
    direction_deg: np.ndarray,
    speed_ms: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    ratio: np.ndarray | None = None,
) -> Samples:
    """dense samples from per-sample farm direction, reference speed and, with measured inflow,
    turbulence ratio, and per-node validity and target; arrays are (S,) and (S, n). wind-frame
    edges are computed once per distinct direction."""
    span = distances(farm.east, farm.north, farm.diameter)
    near = span <= inputs.edge_distance_d
    np.fill_diagonal(near, False)
    speed = (speed_ms / inputs.speed_scale_ms)[:, None]
    scale = inputs.edge_distance_d
    if inputs.edges == "absolute":
        radians = np.deg2rad(direction_deg)[:, None]
        node_inputs = np.concatenate([speed, np.sin(radians), np.cos(radians)], axis=1)
        bearing = np.deg2rad(np.nan_to_num(bearings(farm.east, farm.north)))
        edges = np.stack([span / scale, np.sin(bearing), np.cos(bearing)], axis=-1)[None]
        edge_of = np.zeros(len(direction_deg), dtype=np.int64)
    else:
        node_inputs = speed
        distinct, edge_of = np.unique(direction_deg, return_inverse=True)
        along, across = relations(farm.east, farm.north, distinct, farm.diameter)
        edges = np.stack([along / scale, across / scale], axis=-1)
    if inputs.measured_inflow:
        if ratio is None:
            raise ValueError("a graph with measured inflow needs each sample's turbulence ratio")
        node_inputs = np.concatenate([node_inputs, np.log(ratio)[:, None]], axis=1)
    return Samples(
        inputs=floats(node_inputs),
        edges=floats(edges),
        edge_of=torch.as_tensor(edge_of.ravel(), dtype=torch.int64),
        near=torch.as_tensor(near),
        valid=torch.as_tensor(valid),
        target=floats(np.where(valid, target, 0.0)),
    )


def floats(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32)


@dataclass(frozen=True)
class Slots:
    """where each prepared row sits in the dense samples."""

    sample: np.ndarray
    slot: np.ndarray


def dense_rows(
    rows: pl.DataFrame, farm: Farm, inputs: GraphInputs, folder: Path | None = None
) -> tuple[Samples, Slots]:
    """dense samples of prepared rows; with measured inflow the ratios come from the inflow table
    in folder."""
    check_whole_timestamps(rows)
    group = timestamp_groups(rows)
    slot = rows["turbine_index"].to_numpy().astype(np.int64)
    size = int(group.max(initial=-1)) + 1
    direction = np.zeros(size)
    speed = np.zeros(size)
    direction[group] = rows["farm_direction"].to_numpy()
    speed[group] = rows["reference_speed"].to_numpy()
    ratio = None
    if inputs.measured_inflow:
        if folder is None:
            raise ValueError("a graph with measured inflow reads its ratios from a prepared folder")
        ratio = np.ones(size)
        ratio[group] = inflow.ratios_of(rows, folder)
    valid = np.zeros((size, farm.east.size), dtype=bool)
    target = np.zeros((size, farm.east.size))
    valid[group, slot] = True
    target[group, slot] = rows["y"].to_numpy()
    return samples(farm, inputs, direction, speed, valid, target, ratio), Slots(group, slot)


def device_of(context: FitContext) -> torch.device:
    return torch.device(context.device or "cpu")


def batches(total: int, size: int, generator: torch.Generator) -> Iterator[torch.Tensor]:
    """an endless stream of sample indices in batches: consecutive slices of a seeded
    permutation, reshuffled whenever a pass ends. a batch never spans two passes, so a remainder
    shorter than a batch is skipped; with fewer samples than a batch, each batch is all of them."""
    if total == 0:
        raise ValueError("no training samples")
    size = min(size, total)
    while True:
        order = torch.randperm(total, generator=generator)
        for start in range(0, total - size + 1, size):
            yield order[start : start + size]


def run_rounds(
    network: Network,
    data: Samples,
    settings: GraphSettings,
    rounds: int,
    steps_per_round: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    """AdamW for rounds of steps_per_round optimizer steps, whatever the data size, with the
    learning rate decaying once per round and no early stopping; returns the mean training loss
    of each round."""
    stream = batches(data.valid.shape[0], settings.batch_size, torch.Generator().manual_seed(seed))
    optimiser = torch.optim.AdamW(
        network.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=settings.learning_rate_decay)
    curve = []
    network.train()
    for _ in range(rounds):
        loss_sum = torch.zeros((), device=device)
        weight_sum = torch.zeros((), device=device)
        for _ in range(steps_per_round):
            nodes, edges, adjacency, valid, target = data.take(next(stream).to(device))
            loss = centred_loss(network(nodes, edges, adjacency), target, valid)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            weight = valid.sum()
            loss_sum += loss.detach() * weight
            weight_sum += weight
        schedule.step()
        curve.append(float(loss_sum / weight_sum.clamp(min=1)))
    return curve


def evaluate(network: Network, data: Samples) -> torch.Tensor:
    network.eval()
    out = []
    with torch.no_grad():
        for batch in torch.arange(data.valid.shape[0], device=data.valid.device).split(
            PREDICT_BATCH
        ):
            nodes, edges, adjacency, valid, _ = data.take(batch)
            out.append(centred_output(network(nodes, edges, adjacency), valid))
    return torch.cat(out) if out else torch.zeros((0, data.valid.shape[1]))


@contextmanager
def deterministic(seed: int, threads: int) -> Iterator[None]:
    """deterministic torch kernels, seeded weights and the job's thread count for one fit."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.set_num_threads(previous[1])


class FittedGraph:
    def __init__(
        self,
        network: Network,
        farm: Farm,
        inputs: GraphInputs,
        device: torch.device,
        curve: list[float],
        seconds: float,
        folder: Path | None = None,
    ):
        self.network = network
        self.farm = farm
        self.inputs = inputs
        self.device = device
        self.curve = pl.DataFrame(
            {"round": np.arange(1, len(curve) + 1), "loss": np.asarray(curve, dtype=np.float64)}
        )
        self.seconds = seconds
        self.folder = folder

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        data, where = dense_rows(rows, self.farm, self.inputs, self.folder)
        out = evaluate(self.network, data.to(self.device)).double().cpu().numpy()
        mean = out[where.sample, where.slot]
        return pl.DataFrame({"mean": mean}).with_columns(variance=pl.lit(None, dtype=pl.Float64))

    def state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.network.state_dict().items()}


class Graph:
    """a Daenens-style graph network on the projected loss; initial holds pretrained weights."""

    def __init__(
        self,
        name: str,
        *,
        edges: Edges,
        settings: GraphSettings,
        steps_per_round: int,
        turbine: TurbineConfig,
        edge_distance_d: float,
        speed_scale_ms: float,
        initial: Mapping[str, torch.Tensor] | None = None,
        measured_inflow: bool = False,
    ):
        self.name = name
        self.settings = settings
        self.steps_per_round = steps_per_round
        self.diameter_m = turbine.rotor_diameter_m
        self.inputs = GraphInputs(edges, edge_distance_d, speed_scale_ms, measured_inflow)
        self.initial = initial

    def network(self) -> Network:
        edge_inputs = 3 if self.inputs.edges == "absolute" else 2
        return Network(self.inputs.node_inputs, edge_inputs, self.settings)

    def fit(self, train: pl.DataFrame, context: FitContext) -> FittedGraph:
        """reads layout.parquet, and with measured inflow the inflow table, in prepared_dir;
        trains on context.device, or the CPU."""
        device = device_of(context)
        farm = read_farm(context.prepared_dir, self.diameter_m, train)
        start = time.perf_counter()
        with deterministic(context.seed, context.threads):
            network = self.network()
            if self.initial is not None:
                network.load_state_dict(self.initial)
            network.to(device)
            data, _ = dense_rows(train, farm, self.inputs, context.prepared_dir)
            curve = run_rounds(
                network,
                data.to(device),
                self.settings,
                self.settings.rounds,
                self.steps_per_round,
                context.seed,
                device,
            )
        seconds = time.perf_counter() - start
        return FittedGraph(network, farm, self.inputs, device, curve, seconds, context.prepared_dir)


def pretraining_cases(
    sets: np.ndarray, directions_deg: np.ndarray, speed_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """every combination of operating set, whole-degree direction and speed bin, as the
    (mask, degree, bin) keys physics.phi_keys takes, with each case's operating turbines."""
    s, d, b = np.meshgrid(
        np.arange(len(sets)), np.arange(len(directions_deg)), np.arange(speed_bins), indexing="ij"
    )
    valid = np.asarray(sets, dtype=bool)[s.ravel()]
    masks = (valid * (1 << np.arange(valid.shape[1], dtype=np.int64))).sum(axis=1)
    degrees = np.asarray(directions_deg, dtype=np.int64)[d.ravel()]
    return masks, degrees, b.ravel().astype(np.int64), valid


def contrasts(phi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """phi centered over each case's operating turbines, zero elsewhere: the centered log power
    PyWake gives, since the free-stream term is common to every turbine of a case."""
    level = np.where(valid, phi, 0.0).sum(axis=1, keepdims=True) / valid.sum(axis=1, keepdims=True)
    return np.where(valid, phi - level, 0.0)


def operating_sets(masks: np.ndarray, turbines: int) -> np.ndarray:
    """the distinct operating sets of the masks, plus every turbine operating."""
    distinct = np.unique(np.append(np.asarray(masks, dtype=np.int64), (1 << turbines) - 1))
    return ((distinct[:, None] >> np.arange(turbines)) & 1).astype(bool)


def pretrain(
    graph: Graph,
    contrasts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rounds: int,
    context: FitContext,
) -> FittedGraph:
    """the wind-frame network trained from seed context.seed on PyWake contrasts for a fixed
    number of rounds, no early stopping; its state seeds pretrained_graph's fine-tuning."""
    if graph.inputs.edges != "wind_frame":
        raise ValueError("pretraining uses the wind-frame graph")
    device = device_of(context)
    farm = read_farm(context.prepared_dir, graph.diameter_m)
    start = time.perf_counter()
    with deterministic(context.seed, context.threads):
        network = graph.network().to(device)
        data = samples(farm, graph.inputs, *contrasts)
        curve = run_rounds(
            network,
            data.to(device),
            graph.settings,
            rounds,
            graph.steps_per_round,
            context.seed,
            device,
        )
    return FittedGraph(network, farm, graph.inputs, device, curve, time.perf_counter() - start)


def build_model(
    spec: ModelSpec,
    farm: FarmConfig,
    turbine: TurbineConfig,
    protocol: Protocol,
    *,
    configuration: Mapping[str, Any],
    initial: Mapping[str, torch.Tensor] | None = None,
) -> Graph:
    """a graph network from its spec and the menu configuration its farm's tuning chose;
    pretrained_graph passes the pretrained state as initial."""
    if spec.edges is None:
        raise ValueError(f"{spec.name} names no edges")
    if spec.pretrained != (initial is not None):
        raise ValueError(f"{spec.name}: pretrained weights go with pretrained = true only")
    return Graph(
        spec.name,
        edges=spec.edges,
        settings=GraphSettings.from_menu(configuration),
        steps_per_round=protocol.graph.steps_per_round,
        turbine=turbine,
        edge_distance_d=protocol.processing.validity_distance_d,
        speed_scale_ms=protocol.processing.speed_bin_edges_ms[-1],
        initial=initial,
        measured_inflow=spec.inflow == "measured",
    )
