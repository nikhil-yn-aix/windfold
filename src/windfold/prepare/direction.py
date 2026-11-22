"""step 7: provider northing, per-turbine offsets with jump detection and the farm rotation."""

from datetime import timedelta

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from windfold.prepare.inputs import farm_directions
from windfold.prepare.validity import pair_sectors, wrap180


def apply_corrections(
    rows: pl.DataFrame, corrections: pl.DataFrame, columns: tuple[str, ...]
) -> pl.DataFrame:
    """add each turbine's correction in force at the row's timestamp to the angle columns.

    corrections hold turbine, valid_from and offset_deg; a row takes the latest correction that
    starts at or before it, and rows before a turbine's first correction take that first one.
    a turbine without corrections fails, so no row passes uncorrected.
    """
    missing = set(rows.get_column("turbine").unique()) - set(corrections.get_column("turbine"))
    if missing:
        raise ValueError(f"no northing correction for turbines {sorted(missing)}")
    ordered = corrections.sort("turbine", "valid_from")
    first = ordered.group_by("turbine").agg(pl.col("offset_deg").first().alias("first_offset"))
    joined = (
        rows.with_row_index("row")
        .sort("timestamp")
        .join_asof(
            ordered.sort("valid_from"),
            left_on="timestamp",
            right_on="valid_from",
            by="turbine",
            strategy="backward",
            check_sortedness=False,
        )
        .join(first, on="turbine", how="left")
        .with_columns(pl.coalesce("offset_deg", "first_offset").alias("offset_deg"))
        .sort("row")
    )
    return joined.with_columns([(pl.col(c) + pl.col("offset_deg")) % 360.0 for c in columns]).drop(
        "row", "valid_from", "offset_deg", "first_offset"
    )


def dense_angles(rows: pl.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """the angle column as a timestamps-by-turbines array with nan where a row is absent.

    also returns each row's position in that array.
    """
    stamps = rows.get_column("timestamp").to_physical().to_numpy()
    _, row_of = np.unique(stamps, return_inverse=True)
    turbines = rows.get_column("turbine").to_numpy()
    names, col_of = np.unique(turbines, return_inverse=True)
    angles = np.full((row_of.max() + 1, names.size), np.nan)
    angles[row_of, col_of] = rows.get_column(column).cast(pl.Float64).fill_null(np.nan).to_numpy()
    return angles, row_of, col_of


def consensus_deviations(
    rows: pl.DataFrame, column: str, min_others: int, reference: str | None = None
) -> pl.Series:
    """each row's angle minus the median angle of the other turbines at its timestamp.

    the other turbines' angles come from reference, by default column itself. the median is
    robust to one turbine far off north, which a leave-one-out mean is not. angles are
    unwrapped around the circular mean of the timestamp first; rows with fewer than
    min_others other turbines get null.
    """
    angles, row_of, col_of = dense_angles(rows, column)
    others_angles = angles if reference is None else dense_angles(rows, reference)[0]
    radians = np.deg2rad(others_angles)
    centre = np.rad2deg(np.arctan2(np.nansum(np.sin(radians), 1), np.nansum(np.cos(radians), 1)))
    unwrapped = centre[:, None] + wrap180(others_angles - centre[:, None])
    deviations = np.full(angles.shape, np.nan)
    for turbine in range(angles.shape[1]):
        others = np.delete(unwrapped, turbine, axis=1)
        enough = (np.sum(~np.isnan(others), axis=1) >= min_others) & ~np.isnan(angles[:, turbine])
        if enough.any():
            median = np.nanmedian(others[enough], axis=1)
            deviations[enough, turbine] = wrap180(angles[enough, turbine] - median)
    return pl.Series("deviation", deviations[row_of, col_of]).fill_nan(None)


def trailing_medians(
    times: np.ndarray, values: np.ndarray, window: timedelta, short_window: timedelta, min_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    frame = pl.DataFrame({"t": times, "v": values})
    medians = frame.select(
        pl.col("v")
        .rolling_median_by("t", window, min_samples=min_rows, closed="left")
        .alias("long"),
        pl.col("v")
        .rolling_median_by("t", short_window, min_samples=min_rows, closed="right")
        .alias("short"),
    )
    long = medians.get_column("long").fill_null(np.nan).to_numpy()
    short = medians.get_column("short").fill_null(np.nan).to_numpy()
    return long, short


def turbine_offsets(
    times: np.ndarray,
    values: np.ndarray,
    window: timedelta,
    short_window: timedelta,
    jump_deg: float,
    min_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """causal offsets for one turbine's deviations in time order, and the rows hit by a jump.

    the offset at a row is the median deviation over the window before it, counted only from the
    turbine's latest jump. a jump is declared at the first row whose median over the short window
    ending there differs from the offset by more than jump_deg; the rows of that short window get
    no offset, and counting restarts after them. rows with fewer than min_rows in a window get no
    offset either.
    """
    offsets = np.full(values.size, np.nan)
    jumped = np.zeros(values.size, dtype=bool)
    start = 0
    while start < values.size:
        long, short = trailing_medians(
            times[start:], values[start:], window, short_window, min_rows
        )
        with np.errstate(invalid="ignore"):
            hits = np.flatnonzero(np.abs(wrap180(short - long)) > jump_deg)
        if hits.size == 0:
            offsets[start:] = long
            break
        detected = start + int(hits[0])
        first = start + int(
            np.searchsorted(times[start:], times[detected] - short_window, side="right")
        )
        offsets[start:first] = long[: first - start]
        jumped[first : detected + 1] = True
        start = detected + 1
    return offsets, jumped


def causal_offsets(
    rows: pl.DataFrame,
    deviation: pl.Series,
    window: timedelta,
    short_window: timedelta,
    jump_deg: float,
    min_rows: int,
) -> pl.DataFrame:
    """offset (null where unknown) and jump flag for every row, from its turbine's deviations.

    a row without a deviation of its own takes the offset and flag of the turbine's latest row
    that has one.
    """
    frame = rows.select("turbine", "timestamp").with_columns(deviation.alias("deviation"))
    parts = []
    for _, part in frame.sort("turbine", "timestamp").group_by("turbine", maintain_order=True):
        usable = part.filter(pl.col("deviation").is_not_null())
        offsets, jumped = turbine_offsets(
            usable.get_column("timestamp").to_numpy(),
            usable.get_column("deviation").to_numpy(),
            window,
            short_window,
            jump_deg,
            min_rows,
        )
        parts.append(
            usable.select("turbine", "timestamp").with_columns(
                pl.Series("offset", offsets).fill_nan(None), pl.Series("jump", jumped)
            )
        )
    found = pl.concat(parts).sort("timestamp")
    return (
        frame.with_row_index("row")
        .sort("timestamp")
        .join_asof(found, on="timestamp", by="turbine", strategy="backward", check_sortedness=False)
        .sort("row")
        .select(pl.col("offset").cast(pl.Float64), pl.col("jump").fill_null(False))
    )


def consensus_offsets(
    rows: pl.DataFrame,
    column: str,
    min_others: int,
    window: timedelta,
    short_window: timedelta,
    jump_deg: float,
    min_rows: int,
) -> pl.DataFrame:
    """causal offsets and jump flags of every row's angle from the consensus, in two passes.

    the first pass measures each angle against the median of the other turbines' angles as read.
    one turbine far off north moves that median for every other turbine, so the second pass
    measures each angle as read against the median of the others' angles less their first-pass
    offsets; a turbine whose first-pass offset is unknown enters as read. the offsets and jumps
    returned are the second pass's.
    """
    first = causal_offsets(
        rows,
        consensus_deviations(rows, column, min_others),
        window,
        short_window,
        jump_deg,
        min_rows,
    )
    corrected = (pl.col(column) - pl.lit(first.get_column("offset"))).fill_null(pl.col(column))
    rows = rows.with_columns((corrected % 360.0).alias("without_offset"))
    deviation = consensus_deviations(rows, column, min_others, reference="without_offset")
    return causal_offsets(rows, deviation, window, short_window, jump_deg, min_rows)


def dip(x: np.ndarray, level: float, depth: float, centre: float, width: float) -> np.ndarray:
    return level - depth * np.exp(-0.5 * ((x - centre) / width) ** 2)


def fit_dip(offsets: np.ndarray, ratios: np.ndarray, half_width: float) -> tuple[float, float]:
    """center and depth of a gaussian dip fitted to ratio against offset from the bearing.

    raises RuntimeError where the fit fails or the median ratio lies outside 0 to 2, where no
    wake dip can sit.
    """
    level = float(np.median(ratios))
    if not 0.0 <= level <= 2.0:
        raise RuntimeError(f"median ratio {level:.2f} is outside 0 to 2")
    depth = min(max(level - float(ratios.min()), 0.01), 1.5)
    p0 = [level, depth, float(offsets[np.argmin(ratios)]), 5.0]
    bounds = ([0.0, 0.0, -half_width, 1.0], [2.0, 1.5, half_width, half_width])
    params, _ = curve_fit(dip, offsets, ratios, p0=p0, bounds=bounds)
    return float(params[2]), float(params[1])


def wake_dip_rotation(
    rows: pl.DataFrame,
    layout: pl.DataFrame,
    *,
    distance_d: tuple[float, float],
    speed_ms: tuple[float, float],
    half_width_deg: float,
    min_rows: int,
    min_depth: float,
) -> tuple[float, pl.DataFrame]:
    """the rotation that centers the farm's wake dips on the layout bearings, and the pair fits.

    for every ordered pair within distance_d rotor diameters, the downstream to upstream power
    ratio is binned by 1 degree of farm direction around the bearing from the downstream turbine
    to the upstream one, with the upstream turbine's speed within speed_ms. a gaussian dip is
    fitted to the bin medians of bins with at least min_rows rows; pairs whose dip is shallower
    than min_depth or centered beyond half the search width are left out. the rotation is minus
    the median dip center, so adding it to every direction moves the dips onto the bearings.
    """
    farm = farm_directions(rows, "direction").select("timestamp", "farm_direction")
    power = rows.select("timestamp", "turbine", "power_kw", "speed").join(farm, on="timestamp")
    pairs = pair_sectors(layout, distance_d[1]).filter(pl.col("distance_d") >= distance_d[0])
    fits = []
    for pair in pairs.iter_rows(named=True):
        upstream = power.filter(
            (pl.col("turbine") == pair["source"])
            & pl.col("speed").is_between(*speed_ms, closed="left")
        ).select("timestamp", "farm_direction", pl.col("power_kw").alias("upstream_kw"))
        downstream = power.filter(pl.col("turbine") == pair["target"]).select(
            "timestamp", pl.col("power_kw").alias("downstream_kw")
        )
        binned = (
            upstream.join(downstream, on="timestamp")
            .with_columns(
                (((pl.col("farm_direction") - pair["bearing_deg"] + 180.0) % 360.0) - 180.0).alias(
                    "offset"
                ),
                (pl.col("downstream_kw") / pl.col("upstream_kw")).alias("ratio"),
            )
            .filter(pl.col("offset").abs() <= half_width_deg)
            .group_by((pl.col("offset").floor() + 0.5).alias("bin"))
            .agg(pl.col("ratio").median(), pl.len().alias("rows"))
            .filter(pl.col("rows") >= min_rows)
            .sort("bin")
        )
        if binned.height < 10:
            continue
        try:
            centre, depth = fit_dip(
                binned.get_column("bin").to_numpy(),
                binned.get_column("ratio").to_numpy(),
                half_width_deg,
            )
        except RuntimeError:
            continue
        if depth >= min_depth and abs(centre) <= half_width_deg / 2:
            fits.append(
                pair | {"centre_deg": centre, "depth": depth, "rows": int(binned["rows"].sum())}
            )
    table = pl.DataFrame(fits) if fits else pl.DataFrame()
    if table.is_empty():
        raise ValueError("no turbine pair shows a clear wake dip, so the farm rotation is unknown")
    return -float(np.median(table.get_column("centre_deg").to_numpy())), table
