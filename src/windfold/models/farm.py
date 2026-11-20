"""the farm as every learner sees it: turbine positions in turbine_index order, and the
timestamp each row belongs to."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from windfold.schemas import LAYOUT, conform


@dataclass(frozen=True)
class Farm:
    """turbine positions in turbine_index order: east and north in meters, hub heights in meters."""

    east: np.ndarray
    north: np.ndarray
    hub_height: np.ndarray
    diameter: float


def farm_from_layout(layout: pl.DataFrame, diameter_m: float) -> Farm:
    """the layout's rows, one per turbine in turbine_index order, with the turbine type's rotor."""
    layout = conform(layout, LAYOUT, extra=True)
    return Farm(
        east=layout["x_m"].to_numpy(),
        north=layout["y_m"].to_numpy(),
        hub_height=layout["hub_height_m"].to_numpy(),
        diameter=diameter_m,
    )


def read_farm(prepared_dir: Path, diameter_m: float, rows: pl.DataFrame | None = None) -> Farm:
    """the farm in prepared_dir/layout.parquet; with rows, each row's turbine_index must name
    its turbine there, so positions can never be matched to the wrong turbine."""
    layout = pl.read_parquet(prepared_dir / "layout.parquet")
    if rows is not None:
        names = layout["turbine"].to_list()
        pairs = rows.select("turbine_index", "turbine").unique().rows()
        wrong = sorted(pair for pair in pairs if pair[0] >= len(names) or names[pair[0]] != pair[1])
        if wrong:
            raise ValueError(
                f"layout.parquet does not list these (turbine_index, turbine): {wrong}"
            )
    return farm_from_layout(layout, diameter_m)


def timestamp_groups(rows: pl.DataFrame) -> np.ndarray:
    return np.unique(rows["timestamp"].to_physical().to_numpy(), return_inverse=True)[1].ravel()
