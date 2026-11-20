"""step 9: keep a row only if every turbine that can wake it is in normal operation."""

import numpy as np
import polars as pl


def wrap180(degrees: np.ndarray) -> np.ndarray:
    return (degrees + 180.0) % 360.0 - 180.0


def disturbed_half_width(distance_m: np.ndarray, diameter_m: np.ndarray) -> np.ndarray:
    """half of the IEC 61400-12-1 disturbed sector, 1.3 arctan(2.5 D / L + 0.15) + 10 degrees."""
    return 0.5 * (1.3 * np.degrees(np.arctan(2.5 * diameter_m / distance_m + 0.15)) + 10.0)


def pair_sectors(layout: pl.DataFrame, max_distance_d: float) -> pl.DataFrame:
    """every ordered source and target pair within max_distance_d rotor diameters of the source.

    layout holds turbine, x_m (east), y_m (north) and rotor_diameter_m, one row per turbine in
    turbine_index order. bearing_deg is the direction from the target to the source, clockwise
    from north: the wind direction that carries the source's wake onto the target.
    """
    names = layout.get_column("turbine").to_list()
    x = layout.get_column("x_m").to_numpy()
    y = layout.get_column("y_m").to_numpy()
    diameter = layout.get_column("rotor_diameter_m").to_numpy()
    source, target = np.meshgrid(np.arange(len(names)), np.arange(len(names)), indexing="ij")
    source, target = source.ravel(), target.ravel()
    keep = source != target
    source, target = source[keep], target[keep]
    east, north = x[source] - x[target], y[source] - y[target]
    distance = np.hypot(east, north)
    pairs = pl.DataFrame(
        {
            "source": [names[i] for i in source],
            "target": [names[i] for i in target],
            "source_index": source.astype(np.uint8),
            "target_index": target.astype(np.uint8),
            "distance_d": distance / diameter[source],
            "bearing_deg": np.degrees(np.arctan2(east, north)) % 360.0,
            "half_width_deg": disturbed_half_width(distance, diameter[source]),
        }
    )
    return pairs.filter(pl.col("distance_d") <= max_distance_d)


def disturbing_sources(rows: pl.DataFrame, sectors: pl.DataFrame) -> pl.Series:
    """for each row, one bit per source whose disturbed sector holds the row's farm direction.

    rows hold turbine_index and farm_direction; a row without a farm direction gets no bits.
    """
    direction = rows.get_column("farm_direction").cast(pl.Float64).fill_null(np.nan).to_numpy()
    index = rows.get_column("turbine_index").to_numpy()
    bits = np.zeros(rows.height, dtype=np.uint32)
    positions = {int(i): np.flatnonzero(index == i) for i in np.unique(index)}
    for pair in sectors.iter_rows(named=True):
        at = positions.get(pair["target_index"])
        if at is None:
            continue
        with np.errstate(invalid="ignore"):
            inside = np.abs(wrap180(direction[at] - pair["bearing_deg"])) <= pair["half_width_deg"]
        bits[at[inside]] |= np.uint32(2 ** pair["source_index"])
    return pl.Series("disturbing", bits, dtype=pl.UInt32)
