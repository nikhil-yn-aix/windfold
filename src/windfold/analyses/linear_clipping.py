"""the linear pair without identity_linear's blow-up on held-out sectors.

identity_linear's unconstrained per-turbine direction terms can predict far outside the targets
on directions it never saw. its single-arc predictions are clipped to the 0.1% and 99.9%
quantiles of the farm's development targets, then pooled waked-row skill against no_wake is
compared with kernel_linear's, and the clipped identity's squared error per waked row with
kernel_linear's.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from windfold.analyses.development import FARMS, STUDY, check_paired, single_arcs, waked_rows

QUANTILES = (0.001, 0.999)
MODELS = ("identity_linear", "kernel_linear", "no_wake")


def target_range(rows: pl.DataFrame) -> tuple[float, float]:
    """the QUANTILES of the development targets."""
    y = rows.filter(pl.col("period") == STUDY)["y"].drop_nulls().to_numpy()
    low, high = np.quantile(y, QUANTILES)
    return float(low), float(high)


def linear_clipping(data: Path, runs: Path, farms: Sequence[str] = FARMS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for farm in farms:
        rows = pl.read_parquet(data / "prepared" / farm / "rows.parquet", columns=["period", "y"])
        low, high = target_range(rows)
        sse = dict.fromkeys(("identity_linear", "identity_linear_clipped", *MODELS[1:]), 0.0)
        count = 0
        for split in single_arcs(runs, farm):
            frames = {model: waked_rows(runs, farm, model, split) for model in MODELS}
            check_paired(frames, f"{farm} {split}")
            y = frames["no_wake"]["y"].to_numpy()
            identity = frames["identity_linear"]["mean"].to_numpy()
            count += y.size
            sse["identity_linear_clipped"] += float(np.sum((y - np.clip(identity, low, high)) ** 2))
            for model in MODELS:
                sse[model] += float(np.sum((y - frames[model]["mean"].to_numpy()) ** 2))
        out[farm] = {
            "clip": [low, high],
            "waked_rows": count,
            "skill": {model: 100 * (1 - value / sse["no_wake"]) for model, value in sse.items()},
            "clipped_identity_minus_kernel_mse": (
                sse["identity_linear_clipped"] - sse["kernel_linear"]
            )
            / count,
        }
    return out
