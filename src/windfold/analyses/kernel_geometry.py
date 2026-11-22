"""the learned kernel on the neighbor geometry that occurs: what windfold's shared kernel adds to
the physics on the turbine pairs seen in development data.

the pairs are every operating upstream neighbor inside the kernel window of every operating
turbine, over a sample of development timestamps in the middle speed bin, sorted by time before
sampling so the sample is fixed by its seed. each single-arc fit's kernel is evaluated on those
pairs at that bin. per farm and windfold variant: the median physics weight (beta), the mean and
rms of the kernel averaged over arcs (percent of power), how much the arcs' kernels spread about
that average, absolutely and relative to its rms, and the correlation of the average with
windfold's.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold.analyses.development import FARMS, STUDY, fit_record, single_arcs
from windfold.config import Protocol, load_farm, load_protocol, load_turbine
from windfold.geometry import relations, upstream
from windfold.models.blocks import KernelBasis, kernel_basis
from windfold.models.farm import farm_from_layout

MODELS = (
    "windfold",
    "windfold_constant_inflow",
    "windfold_evidence_scales",
    "windfold_constant_inflow_evidence_scales",
)
SAMPLE = 4000
SAMPLE_SEED = 7


def protocol_basis(protocol: Protocol) -> KernelBasis:
    kernel = protocol.kernel
    return kernel_basis(
        kernel.along_max_d,
        kernel.across_max_d,
        kernel.along_first_knot_d,
        kernel.along_knots,
        kernel.across_first_knot_d,
        kernel.across_knots,
        kernel.degree,
        len(protocol.processing.speed_bin_edges_ms) - 1,
    )


def middle_bin(protocol: Protocol) -> int:
    return (len(protocol.processing.speed_bin_edges_ms) - 1) // 2


def neighbour_pairs(
    rows: pl.DataFrame,
    layout: pl.DataFrame,
    diameter_m: float,
    protocol: Protocol,
    sample: int = SAMPLE,
    seed: int = SAMPLE_SEED,
) -> np.ndarray:
    """the kernel basis, one row per (timestamp, turbine, upstream neighbor) of the sample."""
    farm = farm_from_layout(layout, diameter_m)
    times = (
        rows.select("timestamp", "period", "farm_direction", "speed_bin", "operating_mask")
        .filter(pl.col("period") == STUDY, pl.col("speed_bin") == middle_bin(protocol))
        .unique("timestamp")
        .sort("timestamp")
    )
    times = times.sample(min(times.height, sample), seed=seed)
    along, across = relations(farm.east, farm.north, times["farm_direction"].to_numpy(), diameter_m)
    masks = times["operating_mask"].to_numpy().astype(np.int64)
    operating = ((masks[:, None] >> np.arange(farm.east.size)) & 1).astype(bool)
    both = operating[:, :, None] & operating[:, None, :]
    kernel = protocol.kernel
    inside = both & upstream(along, across, kernel.along_max_d, kernel.across_max_d)
    return protocol_basis(protocol).values(along[inside], across[inside]).toarray()


def kernel_summary(
    design: np.ndarray, records: Sequence[dict[str, Any]], speed_bin: int, per_bin: int
) -> tuple[dict[str, Any], np.ndarray]:
    """one variant's summary over its arcs' fit records, and its kernel averaged over arcs."""
    columns = slice(speed_bin * per_bin, (speed_bin + 1) * per_bin)
    blocks = [record["facts"]["blocks"] for record in records]
    values = np.stack([design @ np.asarray(b["shared_kernel"])[columns] for b in blocks])
    mean = values.mean(axis=0)
    rms = float(np.sqrt(np.mean(mean**2)))
    spread = float(np.sqrt(np.mean((values - mean) ** 2)))
    summary = {
        "arcs": len(records),
        "beta_median": float(np.median([b["physics"][0] for b in blocks])),
        "kernel_mean_percent": 100 * float(mean.mean()),
        "kernel_rms_percent": 100 * rms,
        "arc_to_arc_spread_percent": 100 * spread,
        "arc_to_arc_variability": spread / max(rms, 1e-12),
    }
    return summary, mean


def kernel_geometry(
    data: Path,
    runs: Path,
    farms: Sequence[str] = FARMS,
    configs: Path | None = None,
    models: Sequence[str] = MODELS,
    sample: int = SAMPLE,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """per farm, the number of sampled pairs and each variant's kernel summary over all its
    included single-arc fits, with the correlation of its average kernel with the first
    variant's."""
    protocol = load_protocol(configs)
    per_bin, speed_bin = protocol_basis(protocol).per_bin, middle_bin(protocol)
    out: dict[str, Any] = {}
    for farm in farms:
        turbine = load_turbine(load_farm(farm, configs).turbine_type, configs)
        prepared = data / "prepared" / farm
        design = neighbour_pairs(
            pl.read_parquet(prepared / "rows.parquet"),
            pl.read_parquet(prepared / "layout.parquet"),
            turbine.rotor_diameter_m,
            protocol,
            sample,
            seed,
        )
        splits = single_arcs(runs, farm, waked=False)
        result: dict[str, Any] = {"pairs": int(design.shape[0])}
        means = {}
        for model in models:
            records = [fit_record(runs, farm, model, split) for split in splits]
            result[model], means[model] = kernel_summary(design, records, speed_bin, per_bin)
        first = models[0]
        for model in models[1:]:
            correlation = np.corrcoef(means[model], means[first])[0, 1]
            result[model][f"correlation_with_{first}"] = float(correlation)
        out[farm] = result
    return out
