"""what windfold learns in the wind frame, at the middle speed bin. along the wake centerline of
one upstream turbine, per farm: the calibrated PyWake wake at the farm's reference turbulence
scaled by windfold's learned physics weight, windfold's whole effect of that turbine (physics
weight times PyWake plus the shared kernel) with its range over the sector-held-out fits and the
shared kernel alone; ticks mark the turbine spacings, the only distances the data inform."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from windfold import config, physics
from windfold.models.blocks import KernelBasis
from windfold.models.registry import engine_settings
from windfold.report.sources import Sources
from windfold.report.style import Style

ALONG_D = np.linspace(2.0, 30.0, 113)
MIDDLE_BIN = 1
RANDOM_SPLIT = "random_1"


@dataclass(frozen=True)
class Learned:
    """one farm's centerline profiles, as fractions of log power."""

    physics: np.ndarray
    whole: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    kernel: np.ndarray
    spacings: np.ndarray


def spacings_d(layout: pl.DataFrame, diameter: float) -> np.ndarray:
    """every distance between two turbines, in rotor diameters: the only centerline distances
    the data can inform."""
    east, north = layout["x_m"].to_numpy(), layout["y_m"].to_numpy()
    gaps = np.hypot(east[:, None] - east[None, :], north[:, None] - north[None, :]) / diameter
    return np.sort(gaps[np.triu_indices(len(east), 1)])


def percent(values: np.ndarray) -> np.ndarray:
    """a change in log power as a change in power, in percent."""
    return 100.0 * np.expm1(values)


def centreline_physics(
    family: str,
    parameters: Mapping[str, float],
    turbine: config.TurbineConfig,
    hub_height: float,
    speed_ms: float,
    truncate_sd: float,
    floor_fraction: float,
) -> np.ndarray:
    """phi of a receiver at each ALONG_D straight downwind of one operating turbine, smoothed over
    the family's direction spread as the physics table smooths it."""
    diameter = turbine.rotor_diameter_m
    east = np.zeros(len(ALONG_D) + 1)
    north = np.r_[0.0, -ALONG_D * diameter]
    hub = np.full(len(east), hub_height)
    model = physics.wake_model(family, parameters, physics.pywake_turbine(turbine))
    offsets, weights = physics.spread_weights(parameters[physics.SPREAD], truncate_sd)
    codes = np.full((len(offsets), len(east)), physics.RECEIVING)
    codes[:, 0] = physics.OPERATING
    watts = physics.power(
        model, east, north, hub, codes, offsets % 360, np.full(len(offsets), speed_ms)
    )
    smoothed = weights @ watts
    floor_w = floor_fraction * turbine.rated_power_kw * 1000.0
    reference = np.log(max(float(physics.free_power(model, [speed_ms])[0]), floor_w))
    return np.log(np.maximum(smoothed[1:], floor_w)) - reference


def kernel_profile(basis: KernelBasis, coefficients: Sequence[float]) -> np.ndarray:
    values = basis.values(ALONG_D, np.zeros_like(ALONG_D))
    own = np.asarray(coefficients)[MIDDLE_BIN * basis.per_bin : (MIDDLE_BIN + 1) * basis.per_bin]
    return values @ own


def learned(
    records: Mapping[str, dict],
    physics_fit: dict,
    turbine: config.TurbineConfig,
    layout: pl.DataFrame,
    basis: KernelBasis,
    protocol: config.Protocol,
) -> Learned:
    """profiles over the single-arc fits; the physics line uses the random split's family and
    learned weight."""
    centre = protocol.processing.speed_bin_centres_ms[MIDDLE_BIN]
    hub = float(np.median(layout["hub_height_m"].to_numpy()))
    phis: dict[str, np.ndarray] = {}

    def phi(family: str) -> np.ndarray:
        if family not in phis:
            phis[family] = centreline_physics(
                family,
                physics_fit["parameters"][family],
                turbine,
                hub,
                centre,
                protocol.physics.truncate_sd,
                protocol.physics.power_floor_fraction,
            )
        return phis[family]

    wholes, kernels = [], []
    for split, record in records.items():
        if split == RANDOM_SPLIT:
            continue
        facts = record["facts"]
        kernel = kernel_profile(basis, facts["blocks"]["shared_kernel"])
        weight = facts["blocks"]["physics"][0]
        wholes.append(weight * phi(facts["family"]) + kernel)
        kernels.append(kernel)
    reference = records[RANDOM_SPLIT]["facts"]
    whole = np.vstack(wholes)
    return Learned(
        physics=reference["blocks"]["physics"][0] * phi(reference["family"]),
        whole=whole.mean(axis=0),
        lower=whole.min(axis=0),
        upper=whole.max(axis=0),
        kernel=np.vstack(kernels).mean(axis=0),
        spacings=spacings_d(layout, turbine.rotor_diameter_m),
    )


def profile_panel(ax: Axes, found: Learned, style: Style) -> None:
    """the profiles up to the farm's widest turbine spacing, with a tick at every spacing."""
    lines = style.report.lines
    own = style.role("windfold")
    shown = ALONG_D <= found.spacings.max()
    x = ALONG_D[shown]
    style.zero_line(ax, vertical=False)
    lower, upper = percent(found.lower[shown]), percent(found.upper[shown])
    ax.fill_between(x, lower, upper, color=own, alpha=0.2, lw=0)
    ax.plot(x, percent(found.physics[shown]), color=style.role("physics"), ls="dashed")
    ax.plot(x, percent(found.whole[shown]), color=own, lw=lines.thick_pt)
    ax.plot(x, percent(found.kernel[shown]), color=own, ls="dotted")
    near = found.spacings[found.spacings <= ALONG_D[-1]]
    ax.plot(
        near,
        np.zeros_like(near),
        ls="none",
        marker="|",
        color=style.role("axes"),
        markersize=lines.marker_pt * 1.5,
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xlim(0, ALONG_D[-1])
    ax.set_xticks([0, 10, 20, 30])


def draw(found: Mapping[str, Learned], style: Style) -> Figure:
    farms = list(found)
    lines = style.report.lines
    with style.context():
        rows = (len(farms) + 1) // 2
        figure = style.figure(170.0, 24.0 + 50.0 * rows)
        axes = figure.subplots(rows, 2, sharex=True, sharey=True, squeeze=False).ravel()
        for ax, farm, letter in zip(axes, farms, "abcd", strict=False):
            profile_panel(ax, found[farm], style)
            style.panel_letter(ax, letter, style.farm_name(farm))
        for ax in axes[len(farms) :]:
            ax.set_visible(False)
        axes[0].set_ylim(-110, 15)
        axes[0].set_yticks([-100, -50, 0])
        figure.supylabel("effect on power (%)")
        figure.supxlabel("distance along the wind (D)")
        own = style.role("windfold")
        keys = [
            Line2D([], [], color=style.role("physics"), ls="dashed", label="PyWake × weight"),
            Line2D([], [], color=own, lw=lines.thick_pt, label="windfold"),
            Line2D([], [], color=own, ls="dotted", label="shared correction"),
        ]
        figure.legend(handles=keys, loc="outside upper right", ncols=len(keys))
    return figure


def records_of(folder: Path) -> dict[str, dict]:
    """the single-arc and random-split fit records of one farm's windfold fits."""
    found = {}
    for path in sorted(folder.glob("*.json")):
        split = path.stem
        if split == RANDOM_SPLIT or split.endswith("_single"):
            found[split] = json.loads(path.read_text(encoding="utf-8"))
    return found


def figure(sources: Sources, style: Style) -> Figure:
    protocol = sources.protocol()
    basis = engine_settings(protocol).kernel
    model = protocol.endpoints.early_history_advantage.model
    found = {}
    for farm in style.report.farms.order:
        folder = sources.runs / sources.main / "predictions" / farm / model
        records = records_of(folder)
        if RANDOM_SPLIT not in records or len(records) < 2:
            continue
        farm_config = config.load_farm(farm, sources.configs)
        prepared = sources.data / "prepared" / farm
        found[farm] = learned(
            records,
            json.loads((prepared / "physics_fit.json").read_text(encoding="utf-8")),
            config.load_turbine(farm_config.turbine_type, sources.configs),
            pl.read_parquet(prepared / "layout.parquet"),
            basis,
            protocol,
        )
    return draw(found, style)
