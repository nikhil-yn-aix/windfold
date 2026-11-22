"""step 12: the figures that let a person check one farm's preparation by eye."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.figure import Figure

from windfold.config import Curve, FarmConfig, Processing, TurbineConfig
from windfold.locations import INSPECT_FILES
from windfold.prepare.direction import consensus_deviations
from windfold.prepare.pipeline import Prepared, write_atomic
from windfold.prepare.power_curves import curve_power
from windfold.prepare.validity import pair_sectors, wrap180

GROUPS = {
    "unknown": ("power_missing", "unknown_event"),
    "stopped": (
        "availability",
        "downtime",
        "stop_event",
        "power_floor",
        "state_code",
        "running_time",
        "stop_time",
        "derated",
    ),
    "curtailed or derated": ("curtailment_event", "derate_event", "binding_setpoint"),
    "sensor fault": (
        "speed_missing",
        "direction_missing",
        "out_of_range",
        "frozen_row",
        "stuck_speed",
        "stuck_direction",
        "linear_mean_wind_direction",
        "linear_mean_nacelle_position",
    ),
    "power curve": ("power_curve",),
    "excluded or controlled": ("exclusion", "control", "minutes_per_window"),
    "direction": ("manual_yaw", "direction_jump", "offset_unknown"),
    "iced": ("icing_event",),
    "wake source": ("wake_source",),
    "no reference or outside 5 to 10 m/s": ("reference", "speed_window", "valid_turbines"),
}


def save(figure: Figure, path: Path) -> None:
    """the figure as a pdf without a creation date, so the same rows give the same file."""
    write_atomic(path, lambda p: figure.savefig(p, format="pdf", metadata={"CreationDate": None}))


def windy_week(
    raw: pl.DataFrame, start: datetime, end: datetime, interval: timedelta
) -> tuple[pl.DataFrame, float]:
    """the first monday-to-monday week in [start, end) where every turbine reports power at least
    99% of the time and the median speed is at least 5 m/s, and its share of rows with power.

    without such a week, the most complete week with that median speed; without that, the most
    complete week.
    """
    expected = raw.get_column("turbine").n_unique() * (timedelta(days=7) // interval)
    weekly = (
        raw.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
        .with_columns(pl.col("timestamp").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(
            (pl.col("power_kw").is_not_null().sum() / expected).alias("complete"),
            (pl.col("speed").median() >= 5.0).fill_null(False).alias("windy"),
        )
        .with_columns((pl.col("windy") & (pl.col("complete") >= 0.99)).alias("chosen"))
    )
    if weekly.is_empty():
        raise ValueError(f"no raw rows between {start} and {end}")
    best = weekly.sort(
        ["chosen", "windy", "complete", "week"], descending=[True, True, True, False]
    )
    if best.get_column("chosen")[0]:
        best = weekly.filter(pl.col("chosen")).sort("week")
    week, complete = best.row(0, named=True)["week"], best.row(0, named=True)["complete"]
    rows = raw.filter(
        (pl.col("timestamp") >= week) & (pl.col("timestamp") < week + timedelta(days=7))
    )
    return rows, float(complete)


def raw_week(
    raw: pl.DataFrame, start: datetime, end: datetime, interval: timedelta, path: Path
) -> None:
    week, complete = windy_week(raw, start, end, interval)
    figure = Figure(figsize=(11, 8), layout="constrained")
    axes = figure.subplots(3, 1, sharex=True)
    for name, part in week.sort("timestamp").group_by("turbine", maintain_order=True):
        times = part.get_column("timestamp").to_numpy()
        for ax, column in zip(axes, ("power_kw", "speed", "wind_direction"), strict=True):
            ax.plot(times, part.get_column(column).to_numpy(), lw=0.7, label=str(name[0]))
    for ax, label in zip(
        axes, ("power (kW)", "nacelle speed (m/s)", "wind direction (deg)"), strict=True
    ):
        ax.set_ylabel(label)
    axes[0].legend(ncol=8, fontsize=7)
    first = week.get_column("timestamp").min()
    axes[0].set_title(f"raw rows, week from {first:%Y-%m-%d}, power present in {complete:.1%}")
    save(figure, path)


def status_mix(dropped: pl.DataFrame, kept: pl.DataFrame, path: Path) -> None:
    """share of each turbine's rows inside the windows by the first rule that dropped it."""
    group = {rule: name for name, rules in GROUPS.items() for rule in rules}
    shares = (
        dropped.filter(pl.col("rule") != "windows")
        .with_columns(pl.col("rule").replace_strict(group, default="other").alias("group"))
        .group_by("turbine", "group")
        .agg(pl.col("rows_dropped").sum().alias("rows"))
        .vstack(
            kept.select("turbine", pl.lit("kept").alias("group"), pl.col("rows_kept").alias("rows"))
        )
        .with_columns((pl.col("rows") / pl.col("rows").sum().over("turbine")).alias("share"))
    )
    turbines = sorted(shares.get_column("turbine").unique().to_list())
    order = ["kept", *GROUPS, "other"]
    figure = Figure(figsize=(11, 0.35 * len(turbines) + 2), layout="constrained")
    ax = figure.subplots()
    left = np.zeros(len(turbines))
    for name in order:
        part = shares.filter(pl.col("group") == name)
        lookup = dict(zip(part.get_column("turbine"), part.get_column("share"), strict=True))
        width = np.array([lookup.get(t, 0.0) for t in turbines])
        if width.any():
            ax.barh(turbines, width, left=left, label=name)
            left += width
    ax.set_xlabel("share of the turbine's rows inside the windows")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.invert_yaxis()
    save(figure, path)


def power_curves(
    rows: pl.DataFrame, curve: Curve, rated_kw: float, processing: Processing, path: Path
) -> None:
    turbines = sorted(rows.get_column("turbine").unique().to_list())
    columns = 4
    figure = Figure(figsize=(12, 3 * -(-len(turbines) // columns)), layout="constrained")
    axes = np.atleast_1d(
        figure.subplots(-(-len(turbines) // columns), columns, sharex=True, sharey=True)
    ).ravel()
    speed = np.linspace(0.0, 20.0, 401)
    upper = curve_power(speed * (1 + processing.curve_speed_tolerance), curve)
    lower = curve_power(speed * (1 - processing.curve_speed_tolerance), curve)
    margin = processing.curve_power_tolerance * rated_kw
    for ax, name in zip(axes, turbines, strict=False):
        part = rows.filter(pl.col("turbine") == name)
        ax.hexbin(
            part.get_column("speed").to_numpy(),
            part.get_column("power_kw").to_numpy(),
            gridsize=60,
            bins="log",
            mincnt=1,
            extent=(0, 20, -100, rated_kw * 1.1),
        )
        ax.plot(speed, curve_power(speed, curve), color="black", lw=0.8)
        ax.plot(speed, upper + margin, color="red", lw=0.6, ls="--")
        ax.plot(speed, lower - margin, color="red", lw=0.6, ls="--")
        ax.set_title(f"{name}, {part.height} kept rows", fontsize=8)
    for ax in axes[len(turbines) :]:
        ax.set_visible(False)
    figure.supxlabel("own nacelle speed (m/s)")
    figure.supylabel("power (kW)")
    figure.suptitle(f"kept rows against the curve ({curve.source[:80]}) and its bounds")
    save(figure, path)


def direction_offsets(rows: pl.DataFrame, jumps: list[dict], min_others: int, path: Path) -> None:
    """daily median deviation from the other turbines, as read and as calibrated."""
    raw = consensus_deviations(rows, "direction_raw", min_others)
    calibrated = consensus_deviations(rows, "direction", min_others)
    daily = (
        rows.select("turbine", pl.col("timestamp").dt.truncate("1d").alias("day"))
        .with_columns(raw.alias("raw"), calibrated.alias("calibrated"))
        .group_by("turbine", "day")
        .agg(pl.col("raw").median(), pl.col("calibrated").median())
        .sort("turbine", "day")
    )
    turbines = sorted(daily.get_column("turbine").unique().to_list())
    figure = Figure(figsize=(12, 1.6 * len(turbines) + 1), layout="constrained")
    axes = np.atleast_1d(figure.subplots(len(turbines), 1, sharex=True))
    for ax, name in zip(axes, turbines, strict=True):
        part = daily.filter(pl.col("turbine") == name)
        days = part.get_column("day").to_numpy()
        ax.plot(days, part.get_column("raw").to_numpy(), lw=0.6, label="as read")
        ax.plot(days, part.get_column("calibrated").to_numpy(), lw=0.6, label="calibrated")
        for jump in (j for j in jumps if j["turbine"] == name):
            first, last = (datetime.fromisoformat(str(jump[k])) for k in ("first", "last"))
            ax.axvspan(first, last, color="red", alpha=0.4)
        ax.axhline(0.0, color="black", lw=0.4)
        ax.set_ylabel(name)
    axes[0].legend(fontsize=7, ncol=2)
    figure.supylabel("deviation from the median of the other turbines (deg)")
    figure.suptitle("daily median direction deviation; red: rows dropped around a found jump")
    save(figure, path)


def speed_drift(rows: pl.DataFrame, title: str, path: Path) -> None:
    """monthly median of each free-stream turbine's speed over the reference speed."""
    monthly = (
        rows.filter(~pl.col("waked"))
        .with_columns(
            pl.col("timestamp").dt.truncate("1mo").alias("month"),
            (pl.col("speed") / pl.col("reference_speed")).alias("ratio"),
        )
        .group_by("turbine", "month")
        .agg(pl.col("ratio").median(), pl.len().alias("rows"))
        .filter(pl.col("rows") >= 50)
        .sort("turbine", "month")
    )
    figure = Figure(figsize=(12, 4.5), layout="constrained")
    ax = figure.subplots()
    for name, part in monthly.group_by("turbine", maintain_order=True):
        ax.plot(
            part.get_column("month").to_numpy(),
            part.get_column("ratio").to_numpy(),
            lw=0.8,
            label=str(name[0]),
        )
    ax.axhline(1.0, color="black", lw=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("free-stream speed / reference speed, monthly median")
    if not monthly.is_empty():
        ax.legend(ncol=8, fontsize=7)
    save(figure, path)


def dip_curves(
    rows: pl.DataFrame, pairs: pl.DataFrame, shift: float, half_width: float
) -> pl.DataFrame:
    """per pair, the median downstream over upstream power ratio in 1 degree bins of farm
    direction minus shift, relative to the pair's bearing."""
    power = rows.select("timestamp", "turbine", "power_kw", "farm_direction")
    parts = []
    for pair in pairs.iter_rows(named=True):
        up = power.filter(pl.col("turbine") == pair["source"]).select(
            "timestamp", "farm_direction", pl.col("power_kw").alias("up")
        )
        down = power.filter(pl.col("turbine") == pair["target"]).select(
            "timestamp", pl.col("power_kw").alias("down")
        )
        joined = up.join(down, on="timestamp")
        offset = wrap180(
            joined.get_column("farm_direction").to_numpy() - shift - pair["bearing_deg"]
        )
        parts.append(
            joined.with_columns(
                pl.Series("offset", offset), (pl.col("down") / pl.col("up")).alias("ratio")
            )
            .filter(pl.col("offset").abs() <= half_width)
            .group_by((pl.col("offset").floor() + 0.5).alias("bin"))
            .agg(pl.col("ratio").median())
            .with_columns(pl.lit(f"{pair['source']}-{pair['target']}").alias("pair"))
        )
    return pl.concat(parts)


def wake_dips(
    rows: pl.DataFrame,
    layout: pl.DataFrame,
    diameter_m: float,
    rotation: float,
    processing: Processing,
    path: Path,
) -> None:
    geometry = layout.with_columns(pl.lit(diameter_m).alias("rotor_diameter_m"))
    lowest, highest = processing.dip_distance_d
    pairs = pair_sectors(geometry, highest).filter(pl.col("distance_d") >= lowest)
    half = processing.dip_half_width_deg
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    for ax, shift, title in (
        (axes[0], rotation, "before the farm rotation"),
        (axes[1], 0.0, f"after the farm rotation of {rotation:+.2f} deg"),
    ):
        curves = dip_curves(rows, pairs, shift, half)
        for _, part in curves.sort("bin").group_by("pair", maintain_order=True):
            ax.plot(
                part.get_column("bin").to_numpy(),
                part.get_column("ratio").to_numpy(),
                lw=0.3,
                color="grey",
            )
        median = curves.group_by("bin").agg(pl.col("ratio").median()).sort("bin")
        ax.plot(
            median.get_column("bin").to_numpy(),
            median.get_column("ratio").to_numpy(),
            lw=1.5,
            color="black",
        )
        ax.axvline(0.0, color="red", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("farm direction minus layout bearing (deg)")
    axes[0].set_ylabel("downstream over upstream power, bin median")
    span = f"{lowest:g} to {highest:g} diameters apart"
    figure.suptitle(f"wake dips of {pairs.height} pairs {span}; black: median")
    save(figure, path)


def write_inspect(
    raw: pl.DataFrame,
    prepared: Prepared,
    farm: FarmConfig,
    turbine: TurbineConfig,
    processing: Processing,
    folder: Path,
) -> tuple[str, ...]:
    """every inspect file of one farm into folder, each replaced whole; returns their names.

    raw holds the farm's read rows; the raw week comes from its development window. the curve,
    the jumps, the factors and the farm rotation are the ones in the provenance of prepared.
    """
    folder.mkdir(parents=True, exist_ok=True)
    provenance = prepared.provenance
    used = provenance["power_curve"]
    curve = Curve(
        speed_ms=tuple(used["speed_ms"]), value=tuple(used["power_kw"]), source=used["source"]
    )
    rows = prepared.rows
    kept = rows.group_by("turbine").agg(pl.len().cast(pl.UInt64).alias("rows_kept")).sort("turbine")
    window = farm.windows.development
    interval = timedelta(minutes=farm.rules.interval_minutes)
    raw_week(raw, window.start, window.end, interval, folder / "raw_week.pdf")
    status_mix(prepared.dropped, kept, folder / "status_mix.pdf")
    power_curves(rows, curve, turbine.rated_power_kw, processing, folder / "power_curves.pdf")
    jumps = provenance.get("direction", {}).get("offsets", {}).get("jumps", [])
    direction_offsets(
        rows, jumps, processing.consensus_min_others, folder / "direction_offsets.pdf"
    )
    title = (
        "speeds after each turbine's anemometer factor from the calibration window"
        if provenance["speed_factors"]["turbines"]
        else "speeds as read; no calibration window, so every anemometer factor is 1"
    )
    speed_drift(rows, title, folder / "speed_drift.pdf")
    rotation = provenance["northing"]["farm_rotation_deg"]
    wake_dips(
        rows,
        prepared.layout,
        turbine.rotor_diameter_m,
        rotation,
        processing,
        folder / "wake_dips.pdf",
    )
    write_atomic(folder / "rows_dropped.csv", lambda p: prepared.dropped.write_csv(p))
    write_atomic(folder / "rows_kept.csv", lambda p: kept.write_csv(p))
    return INSPECT_FILES
