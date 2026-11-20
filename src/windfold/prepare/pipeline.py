"""the preparation steps for one farm: raw tables in, prepared rows, ledger and provenance out.

each step evaluates its rules on the rows that reach it. a dropped row is counted once, under
the first rule that flags it, in the order the rules run; a flag that cannot be evaluated drops
the row. lines whose rows are minutes (SMARTEOLE before averaging) say so in their reason.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from windfold.config import (
    Curve,
    FarmConfig,
    FarmRules,
    Processing,
    TurbineConfig,
    Window,
    Windows,
)
from windfold.prepare import control, direction, faults, inputs, operating, power_curves, target
from windfold.prepare.validity import disturbing_sources, pair_sectors
from windfold.schemas import LAYOUT, LEDGER, PERIODS, PREPARED_ROWS, UTC, conform

TEN_MINUTES = timedelta(minutes=10)
MEASURED = (
    "power_kw",
    "speed",
    "speed_std",
    "wind_direction",
    "nacelle_position",
    "ambient_temperature",
)
ANGLES = ("direction", "nacelle_position")

Rule = tuple[int, str, str, pl.Series]


@dataclass
class Prepared:
    rows: pl.DataFrame
    layout: pl.DataFrame
    ledger: pl.DataFrame
    dropped: pl.DataFrame
    provenance: dict


class Ledger:
    def __init__(self, rows: pl.DataFrame) -> None:
        self.lines: list[dict] = []
        self.turbines: list[pl.DataFrame] = []
        self.record(rows, 1, "read", "rows as read, one per turbine and interval", 0)

    def record(self, rows: pl.DataFrame, step: int, rule: str, reason: str, dropped: int) -> None:
        self.lines.append(
            {
                "step": step,
                "rule": rule,
                "reason": reason,
                "rows_dropped": dropped,
                "rows_kept": rows.height,
                "timestamps_kept": rows.get_column("timestamp").n_unique(),
            }
        )

    def apply(self, rows: pl.DataFrame, rules: Sequence[Rule]) -> tuple[pl.DataFrame, pl.Series]:
        """drop every flagged row under its first rule; also return each row's first rule.

        rules may share a name with different reasons; each line counts its own rows.
        """
        if not rules:
            return rows, pl.Series("dropped_by", [None] * rows.height, dtype=pl.String)
        first = pl.lit(None, dtype=pl.UInt32)
        for line, (_, _, _, mask) in reversed(list(enumerate(rules))):
            flagged = pl.lit(mask.fill_null(True))
            first = pl.when(flagged).then(pl.lit(line, dtype=pl.UInt32)).otherwise(first)
        lines = rows.select(first.alias("line")).to_series()
        names = {line: rule for line, (_, rule, _, _) in enumerate(rules)}
        steps = {line: step for line, (step, _, _, _) in enumerate(rules)}
        self.turbines.append(
            rows.select("turbine")
            .with_columns(lines)
            .filter(pl.col("line").is_not_null())
            .group_by("line", "turbine")
            .agg(pl.len().cast(pl.UInt64).alias("rows_dropped"))
            .sort("line", "turbine")
            .select(
                pl.col("line").replace_strict(steps, return_dtype=pl.UInt8).alias("step"),
                pl.col("line").replace_strict(names, return_dtype=pl.String).alias("rule"),
                "turbine",
                "rows_dropped",
            )
        )
        marked = rows.with_columns(lines)
        for line, (step, rule, reason, _) in enumerate(rules):
            before = marked.height
            marked = marked.filter(pl.col("line").ne_missing(line))
            self.record(marked, step, rule, reason, before - marked.height)
        attributed = lines.replace_strict(names, return_dtype=pl.String).alias("dropped_by")
        return marked.drop("line"), attributed

    def frame(self) -> pl.DataFrame:
        return conform(pl.DataFrame(self.lines, schema=LEDGER), LEDGER)

    def by_turbine(self) -> pl.DataFrame:
        """rows each rule dropped, per turbine, in the order the rules ran."""
        empty = pl.DataFrame(
            schema={
                "step": pl.UInt8,
                "rule": pl.String,
                "turbine": pl.String,
                "rows_dropped": pl.UInt64,
            }
        )
        return pl.concat([empty, *self.turbines])


def check_timestamps(rows: pl.DataFrame, interval: timedelta) -> None:
    """refuse rows off the interval grid, outside UTC or repeated for a turbine."""
    if rows.schema["timestamp"] != UTC:
        raise ValueError(f"timestamps must be {UTC}, not {rows.schema['timestamp']}")
    step = interval // timedelta(microseconds=1)
    off_grid = rows.filter(pl.col("timestamp").dt.epoch("us") % step != 0).height
    if off_grid:
        raise ValueError(f"{off_grid} timestamps are not interval starts on the {interval} grid")
    repeated = rows.height - rows.select("turbine", "timestamp").n_unique()
    if repeated:
        raise ValueError(f"{repeated} turbine timestamps appear more than once")


def periods(rows: pl.DataFrame, windows: Windows) -> pl.Series:
    period = pl.lit(None, dtype=pl.String)
    for name in reversed(PERIODS):
        window: Window | None = getattr(windows, name)
        if window is not None:
            inside = (pl.col("timestamp") >= window.start) & (pl.col("timestamp") < window.end)
            period = pl.when(inside).then(pl.lit(name)).otherwise(period)
    return rows.select(period.alias("period")).to_series()


def read_northing(path: Path) -> pl.DataFrame:
    """a provider northing file: a list of [turbine, valid_from in UTC, offset in degrees]."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    return pl.DataFrame(
        {
            "turbine": [str(t) for t, _, _ in entries],
            "valid_from": [v for _, v, _ in entries],
            "offset_deg": [float(o) for _, _, o in entries],
        }
    ).with_columns(pl.col("valid_from").dt.replace_time_zone("UTC").dt.cast_time_unit("us"))


def provider_northing(farm: FarmConfig, raw_dir: Path) -> pl.DataFrame | None:
    """the farm's provider northing corrections, or None when it has none.

    each file applies from its start until the next file's start; at its start every turbine
    takes the correction that file holds at that instant, followed by its later entries.
    """
    entries = farm.northing.provider_corrections
    if not entries:
        return None
    parts = []
    for entry, following in zip(entries, [*entries[1:], None], strict=True):
        corrections = read_northing(raw_dir / entry.file)
        end = following.start if following else None
        in_force = (
            corrections.filter(pl.col("valid_from") <= entry.start)
            .sort("valid_from")
            .group_by("turbine")
            .agg(pl.col("offset_deg").last())
            .select("turbine", pl.lit(entry.start).cast(UTC).alias("valid_from"), "offset_deg")
        )
        later = corrections.filter(pl.col("valid_from") > entry.start)
        if end is not None:
            later = later.filter(pl.col("valid_from") < end)
        parts += [in_force, later]
    return pl.concat(parts).sort("turbine", "valid_from")


def events_of(events: pl.DataFrame, rules: FarmRules, kind: str) -> pl.DataFrame:
    """the events of one kind, from every source that can carry it."""
    messages = {(m.status, m.message): m.kind for m in rules.message_kinds}
    logged = operating.classify_events(
        events.filter(pl.col("source") == "status_log"), rules.status_kinds, messages
    )
    codes = pl.DataFrame(
        {"code": [int(c) for c in rules.alarm_kinds], "kind": list(rules.alarm_kinds.values())},
        schema={"code": pl.Int64, "kind": pl.String},
    )
    alarms = events.filter(pl.col("source") == "alarm_log").join(codes, on="code")
    shutdowns = events.filter(pl.col("source") == "shutdown_log").with_columns(
        pl.lit("stop").alias("kind")
    )
    return pl.concat([logged, alarms, shutdowns]).filter(pl.col("kind") == kind)


def mask(rows: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    return rows.select(expr.fill_null(True)).to_series()


def status_rules(
    rows: pl.DataFrame,
    events: pl.DataFrame,
    rules: FarmRules,
    rated_kw: float,
    processing: Processing,
) -> list[Rule]:
    """step 3, in order: unknown power, provider flags, logged events, binding setpoint, floor."""
    found: list[Rule] = [
        (
            3,
            "power_missing",
            "power not measured, so the status is unknown",
            mask(rows, pl.col("power_kw").is_null()),
        )
    ]
    if rules.availability:
        found.append(
            (
                3,
                "availability",
                "time-based availability below 1 or unknown",
                mask(rows, pl.col(rules.availability) < 1.0),
            )
        )
    if rules.downtime_loss:
        found.append(
            (
                3,
                "downtime",
                "lost production to downtime above 0 or unknown",
                mask(rows, pl.col(rules.downtime_loss) > 0.0),
            )
        )
    if rules.state_codes:
        not_zero = pl.any_horizontal([pl.col(c) != 0 for c in rules.state_codes])
        found.append(
            (3, "state_code", "operating state not 0 throughout, or unknown", mask(rows, not_zero))
        )
    if rules.running_seconds:
        least = rules.min_running_seconds or 0.0
        short = pl.col(rules.running_seconds) < least
        reason = f"running under {least:g} s of the interval, or unknown"
        found.append((3, "running_time", reason, mask(rows, short)))
    if rules.stop_seconds:
        stopped = pl.any_horizontal([pl.col(c) > 0 for c in rules.stop_seconds])
        found.append((3, "stop_time", "stop seconds above 0, or unknown", mask(rows, stopped)))
    for kind, reason in (
        ("stop", "overlaps a logged stop or shutdown"),
        ("curtailment", "overlaps a logged curtailment"),
        ("derate", "overlaps a logged power reduction"),
        ("unknown", "overlaps a logged loss of communication"),
    ):
        overlaps = operating.interval_overlaps(rows, events_of(events, rules, kind), TEN_MINUTES)
        found.append((3, f"{kind}_event", reason, overlaps))
    lowest: pl.Series | None = None
    if rules.setpoint_end:
        lowest = operating.lowest_known_setpoint(rows, rules.setpoint_end, TEN_MINUTES)
    elif rules.setpoints:
        lowest = rows.select(pl.coalesce(rules.setpoints)).to_series()
    if lowest is not None:
        binding = operating.binding_setpoint(
            pl.col("power_kw"),
            pl.lit(lowest),
            rated_kw,
            processing.setpoint_binding_fraction,
            processing.setpoint_tolerance_fraction,
        )
        found.append(
            (3, "binding_setpoint", "a setpoint below rated holds the power", mask(rows, binding))
        )
    floor = operating.below_power_floor(
        pl.col("power_kw"), rated_kw, processing.stopped_power_fraction
    )
    found.append(
        (3, "power_floor", "power below the floor, so the turbine is stopped", mask(rows, floor))
    )
    return found


def fault_rules(
    rows: pl.DataFrame,
    rules: FarmRules,
    rated_kw: float,
    processing: Processing,
) -> list[Rule]:
    """step 4: missing, impossible, frozen and stuck measurements, and linear angle means."""
    impossible = (
        (pl.col("speed") < 0.0)
        | (pl.col("speed") > processing.max_speed_ms)
        | (pl.col("wind_direction") < 0.0)
        | (pl.col("wind_direction") >= 360.0)
        | (pl.col("power_kw") > processing.max_power_fraction * rated_kw)
    )
    repeats = processing.stuck_repeats
    found: list[Rule] = [
        (4, "speed_missing", "nacelle speed not measured", mask(rows, pl.col("speed").is_null())),
        (
            4,
            "direction_missing",
            "wind direction not measured",
            mask(rows, pl.col("wind_direction").is_null()),
        ),
        (
            4,
            "out_of_range",
            "speed, direction or power outside its possible range",
            mask(rows, impossible),
        ),
        (
            4,
            "frozen_row",
            "every measurement repeats the previous row",
            faults.frozen_rows(rows, MEASURED, "speed", processing.frozen_min_speed_ms),
        ),
        (
            4,
            "stuck_speed",
            f"{repeats} or more equal speeds in a row",
            faults.stuck_values(rows, "speed", repeats),
        ),
        (
            4,
            "stuck_direction",
            f"{repeats} or more equal wind directions in a row",
            faults.stuck_values(rows, "wind_direction", repeats),
        ),
    ]
    for mean, minimum, maximum in rules.linear_mean_angles:
        linear = faults.linear_angle_mean(
            pl.col(mean),
            pl.col(minimum),
            pl.col(maximum),
            processing.linear_mean_north_deg,
            processing.linear_mean_distance_deg,
        )
        found.append(
            (
                4,
                f"linear_mean_{mean}",
                f"{mean} crossed north but its mean is far from it",
                mask(rows, linear),
            )
        )
    return found


def floats(rows: pl.DataFrame, column: str) -> np.ndarray:
    return rows.get_column(column).cast(pl.Float64).fill_null(np.nan).to_numpy()


def minute_windows(
    rows: pl.DataFrame,
    events: pl.DataFrame,
    rules: FarmRules,
    processing: Processing,
    ledger: Ledger,
) -> pl.DataFrame:
    """drop derated and controlled minutes, then average the rest into 10-minute windows."""
    if rules.derated:
        derated = mask(rows, pl.col(rules.derated).cast(pl.Boolean))
        rows, _ = ledger.apply(
            rows, [(3, "derated", "derated or unknown minute (minutes)", derated)]
        )
    washout = timedelta(minutes=processing.control_washout_minutes)
    controlled = control.with_washout(events.filter(pl.col("source") == "control_log"), washout)
    reason = (
        f"active or unknown farm control or the {processing.control_washout_minutes} minutes"
        " after it (minutes)"
    )
    minute = timedelta(minutes=rules.interval_minutes)
    overlaps = operating.interval_overlaps(rows, controlled, minute, farm_wide=True)
    rows, _ = ledger.apply(rows, [(6, "control", reason, overlaps)])
    started = rows.select("turbine", pl.col("timestamp").dt.truncate(TEN_MINUTES)).n_unique()
    windows = control.interval_means(rows, TEN_MINUTES, processing.control_min_valid_minutes)
    minimum = processing.control_min_valid_minutes
    ledger.record(
        windows,
        6,
        "minutes_per_window",
        f"10-minute windows with under {minimum} complete minutes (rows are windows from here)",
        started - windows.height,
    )
    return windows


def jump_runs(rows: pl.DataFrame) -> list[dict]:
    """the first and last timestamp of each run of rows flagged as a direction jump."""
    ordered = rows.select("turbine", "timestamp", "jump").sort("turbine", "timestamp")
    starts = (pl.col("jump") != pl.col("jump").shift(1)) | (
        pl.col("turbine") != pl.col("turbine").shift(1)
    )
    runs = (
        ordered.with_columns(starts.fill_null(True).cum_sum().alias("run"))
        .filter(pl.col("jump"))
        .group_by("run")
        .agg(
            pl.col("turbine").first(),
            pl.col("timestamp").min().alias("first"),
            pl.col("timestamp").max().alias("last"),
        )
        .sort("turbine", "first")
    )
    return runs.select("turbine", "first", "last").to_dicts()


def calibrate_directions(
    rows: pl.DataFrame,
    events: pl.DataFrame,
    rules: FarmRules,
    processing: Processing,
    northing: pl.DataFrame | None,
    ledger: Ledger,
) -> tuple[pl.DataFrame, dict]:
    """step 7 without the farm rotation: manual yaw, provider northing and causal offsets."""
    rows = rows.with_columns(pl.col("wind_direction").alias("direction"))
    yawed = operating.interval_overlaps(rows, events_of(events, rules, "manual_yaw"), TEN_MINUTES)
    rows, _ = ledger.apply(
        rows, [(7, "manual_yaw", "manual yaw, so the direction is invalid", yawed)]
    )
    record: dict = {}
    if northing is not None:
        rows = direction.apply_corrections(rows, northing, ANGLES)
        record["provider_corrections"] = northing.height
    if rules.causal_offsets and processing.direction_calibration:
        found = direction.consensus_offsets(
            rows,
            "direction",
            processing.consensus_min_others,
            timedelta(days=processing.offset_window_days),
            timedelta(hours=processing.offset_short_window_hours),
            processing.offset_jump_deg,
            processing.offset_min_rows,
        )
        rows = rows.with_columns(found.get_columns())
        record["offsets"] = {
            "consensus": "the median of the other turbines less their own offsets",
            "window_days": processing.offset_window_days,
            "short_window_hours": processing.offset_short_window_hours,
            "jump_deg": processing.offset_jump_deg,
            "jumps": jump_runs(rows),
        }
        rows, _ = ledger.apply(
            rows,
            [
                (
                    7,
                    "direction_jump",
                    "inside the window where a north jump was found",
                    rows.get_column("jump"),
                ),
                (
                    7,
                    "offset_unknown",
                    "too few earlier rows for a direction offset",
                    rows.get_column("offset").is_null(),
                ),
            ],
        )
        rows = rows.with_columns([(pl.col(c) - pl.col("offset")) % 360.0 for c in ANGLES]).drop(
            "offset", "jump"
        )
    return rows, record


def rotate_farm(
    rows: pl.DataFrame,
    layout: pl.DataFrame,
    farm: FarmConfig,
    processing: Processing,
) -> tuple[pl.DataFrame, dict]:
    """step 7's farm rotation, fitted on wake dips of the calibration window and added to all."""
    if farm.northing.farm_rotation != "wake_dips" or not processing.direction_calibration:
        return rows, {"farm_rotation_deg": 0.0}
    window = farm.windows.calibration
    if window is None:
        raise ValueError(f"{farm.name} has no calibration window to fit its farm rotation on")
    rotation, pairs = direction.wake_dip_rotation(
        rows.filter(pl.col("period") == "calibration"),
        layout,
        distance_d=processing.dip_distance_d,
        speed_ms=processing.dip_speed_ms,
        half_width_deg=processing.dip_half_width_deg,
        min_rows=processing.dip_min_rows,
        min_depth=processing.dip_min_depth,
    )
    rows = rows.with_columns([(pl.col(c) + rotation) % 360.0 for c in ANGLES])
    record = {
        "farm_rotation_deg": rotation,
        "window": [window.start, window.end],
        "pairs": pairs.select(
            "source", "target", "distance_d", "bearing_deg", "centre_deg", "depth", "rows"
        ).to_dicts(),
    }
    return rows, record


def operating_wakes(
    rows: pl.DataFrame, context: pl.DataFrame, sectors: pl.DataFrame
) -> pl.DataFrame:
    """rows (timestamp, turbine_index) at timestamps with a farm direction, flagged waked where an
    operating turbine within the sector window wakes them."""
    joined = rows.join(context, on="timestamp")
    return joined.with_columns(disturbing_sources(joined, sectors)).with_columns(
        ((pl.col("disturbing") & pl.col("operating_mask")) != 0).alias("waked")
    )


def low_wind_stops(before: pl.DataFrame, first: pl.Series, rules: FarmRules) -> pl.DataFrame:
    """the rows step 3 dropped as stopped at low wind, as timestamp and turbine_index.

    first is each row's first step-3 rule. a row counts if the power floor dropped it, or if the
    state code dropped it and every state of the interval is 0 or one of the farm's low-wind
    states.
    """
    stopped = before.select(pl.lit(first).eq("power_floor").alias("stopped")).to_series()
    if rules.state_codes and rules.low_wind_states:
        allowed = [0.0, *(float(state) for state in rules.low_wind_states)]
        quiet = pl.all_horizontal([pl.col(c).is_in(allowed) for c in rules.state_codes])
        waiting = before.select((pl.lit(first).eq("state_code") & quiet).alias("w")).to_series()
        stopped = stopped | waiting
    return before.filter(stopped.fill_null(False)).select("timestamp", "turbine_index")


def waked_low_wind_stops(
    stopped: pl.DataFrame,
    kept: pl.DataFrame,
    context: pl.DataFrame,
    sectors: pl.DataFrame,
    edges: Sequence[float],
    what: str,
) -> list[dict]:
    """per speed bin, the stopped rows (low_wind_stops) that an operating turbine wakes.

    context holds, per timestamp, the farm direction, operating mask and reference speed; only
    timestamps that stay in the prepared rows count. what says which rows count as stopped.
    """
    lost = (
        operating_wakes(
            stopped.join(kept.select("timestamp").unique(), on="timestamp"), context, sectors
        )
        .filter(pl.col("waked"))
        .with_columns(inputs.speed_bins(pl.col("reference_speed"), edges).alias("speed_bin"))
    )
    return [
        {
            "step": 12,
            "rule": "waked_low_wind_stop",
            "reason": (
                f"waked rows {what} at reference speeds {edges[k]:g} to"
                f" {edges[k + 1]:g} m/s, counted, already dropped at step 3"
            ),
            "rows_dropped": lost.filter(pl.col("speed_bin") == k).height,
        }
        for k in range(len(edges) - 1)
    ]


def apply_speed_factors(rows: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """every nacelle speed and its spread times its turbine's factor; a turbine without one
    keeps its speeds."""
    return (
        rows.join(
            factors.select("turbine", "factor"), on="turbine", how="left", maintain_order="left"
        )
        .with_columns([pl.col(c) * pl.col("factor").fill_null(1.0) for c in ("speed", "speed_std")])
        .drop("factor")
    )


def temperatures_in_range(
    rows: pl.DataFrame, low_c: float, high_c: float
) -> tuple[pl.DataFrame, int]:
    """rows with every ambient temperature outside [low_c, high_c] made missing, and how many."""
    inside = pl.col("ambient_temperature").is_between(low_c, high_c)
    outside = rows.select((~inside).fill_null(False).sum()).item()
    return rows.with_columns(pl.when(inside).then(pl.col("ambient_temperature"))), int(outside)


def prepare(
    raw: pl.DataFrame,
    events: pl.DataFrame,
    layout: pl.DataFrame,
    farm: FarmConfig,
    turbine: TurbineConfig,
    processing: Processing,
    northing: pl.DataFrame | None = None,
) -> Prepared:
    """run every step on one farm's raw rows and events.

    layout follows the layout schema, one row per turbine in turbine_index order; every rotor
    diameter is the turbine config's. northing holds the provider corrections (provider_northing)
    where the farm has them. a farm with a calibration window is prepared twice: the first pass,
    on the rows before the window ends, gives the free-stream rows that fit each turbine's
    anemometer factor (inputs.speed_factors); the second pass, on all rows, takes the reference
    speed from the free-stream speeds times their factors. every other use of a speed, and the
    speed column, keeps the measured speed. a farm without a calibration window keeps factors
    of 1.
    """
    window = farm.windows.calibration
    settings = (layout, farm, turbine, processing, northing)
    if window is None:
        prepared = prepare_with_factors(raw, events, *settings, None)
        prepared.provenance["speed_factors"] = {
            "fitted_on": None,
            "note": "no calibration window, so every factor is 1",
            "turbines": [],
        }
        return prepared
    first = prepare_with_factors(
        raw.filter(pl.col("timestamp") < window.end), events, *settings, None
    )
    factors = inputs.speed_factors(
        first.rows.filter(pl.col("period") == "calibration"),
        processing.speed_factor_min_timestamps,
    )
    prepared = prepare_with_factors(raw, events, *settings, factors)
    prepared.provenance["speed_factors"] = {
        "fitted_on": [window.start, window.end],
        "min_timestamps": processing.speed_factor_min_timestamps,
        "definition": (
            "median, over calibration timestamps where the turbine and at least one other are"
            " free-stream, of the other free-stream turbines' median speed over its own"
        ),
        "applied_to": "the reference speed only",
        "turbines": factors.to_dicts(),
    }
    return prepared


def prepare_with_factors(
    raw: pl.DataFrame,
    events: pl.DataFrame,
    layout: pl.DataFrame,
    farm: FarmConfig,
    turbine: TurbineConfig,
    processing: Processing,
    northing: pl.DataFrame | None,
    factors: pl.DataFrame | None,
) -> Prepared:
    """every step once; where factors are given, the reference speed comes from the free-stream
    speeds times their factors."""
    layout = conform(layout, LAYOUT)
    geometry = layout.with_columns(pl.lit(turbine.rotor_diameter_m).alias("rotor_diameter_m"))
    rules = farm.rules
    interval = timedelta(minutes=rules.interval_minutes)
    rated = turbine.rated_power_kw
    check_timestamps(raw, interval)
    names = layout.get_column("turbine").to_list()
    strangers = sorted(set(raw.get_column("turbine").unique()) - set(names))
    if strangers:
        raise ValueError(f"turbines {strangers} are not in the layout")
    index = pl.DataFrame(
        {"turbine": names, "turbine_index": list(range(len(names)))},
        schema={"turbine": pl.String, "turbine_index": pl.UInt8},
    )
    rows = raw.join(index, on="turbine", how="left", maintain_order="left")
    low_c, high_c = processing.min_temperature_c, processing.max_temperature_c
    rows, nulled = temperatures_in_range(rows, low_c, high_c)
    ledger = Ledger(rows)
    provenance: dict = {
        "farm": farm.name,
        "turbine_type": farm.turbine_type,
        "processing": processing.model_dump(mode="json"),
        "temperature": {"range_c": [low_c, high_c], "readings_made_missing": nulled},
    }

    rows = rows.with_columns(periods(rows, farm.windows))
    rows, _ = ledger.apply(
        rows, [(2, "windows", "outside the farm's windows", rows.get_column("period").is_null())]
    )
    if interval != TEN_MINUTES:
        rows = minute_windows(rows, events, rules, processing, ledger)
        rows = rows.join(index, on="turbine", how="left", maintain_order="left")
        rows = rows.with_columns(periods(rows, farm.windows))
    rows = rows.with_columns(pl.col("wind_direction").alias("direction_raw"))

    before = rows
    rows, attributed = ledger.apply(rows, status_rules(rows, events, rules, rated, processing))
    stopped = low_wind_stops(before, attributed, rules)
    masks = operating.operating_masks(rows)

    rows, _ = ledger.apply(rows, fault_rules(rows, rules, rated, processing))

    curve: Curve = turbine.power_curve
    if rules.fit_power_curve:
        window = farm.windows.calibration
        if window is None:
            raise ValueError(f"{farm.name} has no calibration window to fit its power curve on")
        fit_rows = rows.filter(pl.col("period") == "calibration")
        curve = power_curves.fit_power_curve(
            floats(fit_rows, "speed"),
            floats(fit_rows, "power_kw"),
            processing.curve_bin_ms,
            processing.curve_min_rows,
            f"fitted on the calibration window {window.start:%Y-%m-%d} to {window.end:%Y-%m-%d}",
        )
    provenance["power_curve"] = {
        "source": curve.source,
        "speed_ms": curve.speed_ms,
        "power_kw": curve.value,
    }
    outliers = power_curves.outside_curve_bounds(
        floats(rows, "speed"),
        floats(rows, "power_kw"),
        curve,
        rated,
        processing.curve_speed_tolerance,
        processing.curve_power_tolerance,
    )
    reason = "power outside the bounds around the power curve"
    rows, _ = ledger.apply(rows, [(5, "power_curve", reason, pl.Series(outliers))])

    excluded = control.exclusion_reasons(rows, farm.exclusions, TEN_MINUTES)
    reasons = list(dict.fromkeys(w.reason or "excluded period" for w in farm.exclusions))
    rows, _ = ledger.apply(rows, [(6, "exclusion", r, excluded.eq_missing(r)) for r in reasons])
    provenance["exclusions"] = [w.model_dump(mode="json") for w in farm.exclusions]

    rows, record = calibrate_directions(rows, events, rules, processing, northing, ledger)
    provenance["direction"] = record

    iced = operating.interval_overlaps(rows, events_of(events, rules, "icing"), TEN_MINUTES)
    rows, _ = ledger.apply(rows, [(8, "icing_event", "overlaps a logged icing event", iced)])

    rows, record = rotate_farm(rows, geometry, farm, processing)
    provenance["northing"] = record

    farm_direction = inputs.farm_directions(rows, "direction")
    sectors = pair_sectors(geometry, processing.validity_distance_d)
    rows = rows.join(farm_direction, on="timestamp", how="left", maintain_order="left")
    rows = rows.join(masks, on="timestamp", how="left", maintain_order="left")
    rows = rows.with_columns(disturbing_sources(rows, sectors))
    invalid = mask(rows, (pl.col("disturbing") & ~pl.col("operating_mask")) != 0)
    within = processing.validity_distance_d
    reason = f"a turbine within {within:g} diameters upwind is not operating normally"
    rows, _ = ledger.apply(rows, [(9, "wake_source", reason, invalid)])
    rows = rows.with_columns((pl.col("disturbing") != 0).alias("waked"))

    on_one_basis = rows if factors is None else apply_speed_factors(rows, factors)
    reference = inputs.reference_speeds(on_one_basis)
    rows = rows.join(reference, on="timestamp", how="left", maintain_order="left")
    edges = processing.speed_bin_edges_ms
    rows = rows.with_columns(inputs.speed_bins(pl.col("reference_speed"), edges).alias("speed_bin"))
    rows, _ = ledger.apply(
        rows,
        [
            (
                10,
                "reference",
                "no free-stream turbine to give a reference speed",
                rows.get_column("reference_speed").is_null(),
            ),
            (
                10,
                "speed_window",
                f"reference speed outside {edges[0]:g} to {edges[-1]:g} m/s",
                rows.get_column("speed_bin").is_null(),
            ),
        ],
    )
    steady = inputs.steady_flags(
        farm_direction, TEN_MINUTES, processing.steady_change_deg, processing.steady_resultant
    )
    rows = rows.join(steady, on="timestamp", how="left", maintain_order="left").with_columns(
        inputs.direction_buckets(pl.col("farm_direction"), processing.direction_bucket_deg).alias(
            "direction_bucket"
        )
    )

    few = mask(rows, pl.len().over("timestamp") < processing.min_valid_turbines)
    reason = f"fewer than {processing.min_valid_turbines} valid turbines at the timestamp"
    rows, _ = ledger.apply(rows, [(11, "valid_turbines", reason, few)])
    rows = target.centred_log_power(rows)

    context = farm_direction.join(masks, on="timestamp").join(reference, on="timestamp")
    what = "stopped below the power floor"
    if rules.state_codes and rules.low_wind_states:
        what += f" or in low-wind states {', '.join(rules.low_wind_states)}"
    for line in waked_low_wind_stops(stopped, rows, context, sectors, edges, what):
        ledger.lines.append(
            line
            | {"rows_kept": rows.height, "timestamps_kept": rows.get_column("timestamp").n_unique()}
        )

    prepared = (
        rows.rename({"ambient_temperature": "temperature"})
        .select(PREPARED_ROWS.names())
        .sort("timestamp", "turbine_index")
    )
    return Prepared(
        conform(prepared, PREPARED_ROWS), layout, ledger.frame(), ledger.by_turbine(), provenance
    )


def write_atomic(path: Path, write) -> None:
    partial = path.with_name(path.name + ".partial")
    write(partial)
    os.replace(partial, path)


def write_prepared(prepared: Prepared, folder: Path) -> None:
    """write rows.parquet, layout.parquet, ledger.csv and provenance.json, each replaced whole
    or not at all."""
    folder.mkdir(parents=True, exist_ok=True)
    write_atomic(
        folder / "rows.parquet", lambda p: prepared.rows.write_parquet(p, compression="zstd")
    )
    write_atomic(folder / "layout.parquet", lambda p: prepared.layout.write_parquet(p))
    write_atomic(folder / "ledger.csv", lambda p: prepared.ledger.write_csv(p))
    text = json.dumps(prepared.provenance, indent=2, default=str)
    write_atomic(folder / "provenance.json", lambda p: p.write_text(text, encoding="utf-8"))
