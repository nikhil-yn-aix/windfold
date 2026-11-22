"""data jobs: read provider files, prepare rows, calibrate physics, write manifests."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from windfold import config, physics, readers, splits
from windfold.config import folder_of, for_farm
from windfold.locations import kernel_folder, manifest_file, prepared_folder, prepared_name
from windfold.prepare import pipeline


def layout_arrays(rows: pl.DataFrame, layout: pl.DataFrame) -> tuple[np.ndarray, ...]:
    order = rows.select("turbine", "turbine_index").unique().sort("turbine_index")
    if order["turbine_index"].to_list() != list(range(order.height)):
        raise ValueError("turbine_index must run 0 to n - 1 over the prepared turbines")
    placed = order.join(layout, on="turbine", how="left", validate="1:1")
    if placed["x_m"].null_count():
        missing = placed.filter(pl.col("x_m").is_null())["turbine"].to_list()
        raise ValueError(f"turbines without a layout entry: {missing}")
    return tuple(placed[c].to_numpy() for c in ("x_m", "y_m", "hub_height_m"))


def read_tables(farm: str, data_dir: str, configs: str | None, out_dir: Path) -> None:
    farm_config = config.load_farm(farm, folder_of(configs))
    readers.write_tables(farm_config, Path(data_dir) / "raw" / farm, out_dir / "tables" / farm)


def prepare_rows(
    farm: str, sensitivity: str | None, data_dir: str, configs: str | None
) -> tuple[pl.DataFrame, pipeline.Prepared]:
    protocol, farm_config, turbine = for_farm(farm, sensitivity, folder_of(configs))
    tables = Path(data_dir) / "tables" / farm
    raw = pl.read_parquet(tables / "rows.parquet")
    prepared = pipeline.prepare(
        raw,
        pl.read_parquet(tables / "events.parquet"),
        pl.read_parquet(tables / "layout.parquet"),
        farm_config,
        turbine,
        protocol.processing,
        pipeline.provider_northing(farm_config, Path(data_dir) / "raw" / farm),
    )
    return raw, prepared


def prepare_farm(
    farm: str, sensitivity: str | None, data_dir: str, configs: str | None, out_dir: Path
) -> None:
    _, prepared = prepare_rows(farm, sensitivity, data_dir, configs)
    pipeline.write_prepared(prepared, out_dir / "prepared" / prepared_name(farm, sensitivity))


def inspect_farm(farm: str, data_dir: str, configs: str | None, out_dir: Path) -> None:
    """every inspect figure of a farm, from preparing it again; the rows must come out exactly
    as stored, so preparation is checked to be deterministic on every run."""
    from windfold.prepare import inspect

    protocol, farm_config, turbine = for_farm(farm, None, folder_of(configs))
    raw, prepared = prepare_rows(farm, None, data_dir, configs)
    stored = pl.read_parquet(Path(data_dir) / "prepared" / farm / "rows.parquet")
    if not prepared.rows.equals(stored):
        raise ValueError(f"preparing {farm} again gave rows that differ from the stored ones")
    inspect.write_inspect(
        raw, prepared, farm_config, turbine, protocol.processing, out_dir / "inspect" / farm
    )


def fit_physics(
    farm: str,
    data_dir: str,
    configs: str | None,
    seed: int,
    workers: int,
    calibrated_farms: list[str],
    out_dir: Path,
) -> None:
    protocol, farm_config, turbine = for_farm(farm, None, folder_of(configs))
    prepared = Path(data_dir) / "prepared"
    if farm_config.windows.calibration is None:
        fits = {
            other: json.loads((prepared / other / "physics_fit.json").read_text())
            for other in calibrated_farms
        }
        fit = physics.fit_from_farms(fits, protocol.physics.defaults)
    else:
        rows = pl.read_parquet(prepared / farm / "rows.parquet")
        east, north, hub = layout_arrays(rows, pl.read_parquet(prepared / farm / "layout.parquet"))
        fit = physics.fit_farm(
            rows.filter(pl.col("period") == "calibration"),
            turbine,
            east,
            north,
            hub,
            np.asarray(protocol.processing.speed_bin_centres_ms),
            protocol.physics.search,
            protocol.physics.defaults,
            protocol.physics.trials,
            seed,
            protocol.physics.truncate_sd,
            protocol.physics.power_floor_fraction,
            workers,
        )
    target = out_dir / "prepared" / farm / "physics_fit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(fit, indent=2, sort_keys=True, default=str))


def physics_table(
    farm: str,
    sensitivity: str | None,
    data_dir: str,
    configs: str | None,
    workers: int,
    out_dir: Path,
) -> None:
    protocol, _, turbine = for_farm(farm, sensitivity, folder_of(configs))
    prepared = Path(data_dir) / "prepared"
    fit = json.loads((prepared / farm / "physics_fit.json").read_text())
    folder = prepared / prepared_name(farm, sensitivity)
    rows = pl.read_parquet(folder / "rows.parquet")
    east, north, hub = layout_arrays(rows, pl.read_parquet(folder / "layout.parquet"))
    table = physics.physics_table(
        rows,
        turbine,
        east,
        north,
        hub,
        np.asarray(protocol.processing.speed_bin_centres_ms),
        fit["parameters"],
        protocol.physics.defaults,
        protocol.physics.truncate_sd,
        protocol.physics.power_floor_fraction,
        workers,
    )
    target = out_dir / "prepared" / prepared_name(farm, sensitivity) / "physics.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(target, compression="zstd")


def inflow_physics(
    farm: str,
    sensitivity: str | None,
    data_dir: str,
    configs: str | None,
    workers: int,
    calibrated_farms: list[str],
    out_dir: Path,
) -> None:
    """the farm's inflow tables, measured and its controls, and record (windfold.inflow); a farm
    without a calibration window takes the median of the calibrated farms' references."""
    from windfold import inflow

    protocol, farm_config, turbine = for_farm(farm, sensitivity, folder_of(configs))
    prepared = Path(data_dir) / "prepared"
    fit = json.loads((prepared / farm / "physics_fit.json").read_text())
    folder = prepared / prepared_name(farm, sensitivity)
    rows = pl.read_parquet(folder / "rows.parquet")
    if farm_config.windows.calibration is None:
        references = {other: inflow.read_reference(prepared / other) for other in calibrated_farms}
        reference = float(np.median(list(references.values())))
        source = "median of " + ", ".join(references)
    else:
        reference, source = inflow.reference(rows), "calibration window"
    east, north, hub = layout_arrays(rows, pl.read_parquet(folder / "layout.parquet"))
    tables = inflow.tables(
        rows,
        turbine,
        east,
        north,
        hub,
        np.asarray(protocol.processing.speed_bin_centres_ms),
        fit["parameters"],
        protocol.physics.defaults,
        protocol.physics.truncate_sd,
        protocol.physics.power_floor_fraction,
        reference,
        protocol.inflow,
        workers,
    )
    target = out_dir / "prepared" / prepared_name(farm, sensitivity)
    target.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(target / inflow.TABLES[name], compression="zstd")
    note = inflow.record(rows, reference, source, inflow.ratio_grid(protocol.inflow))
    (target / inflow.RECORD).write_text(json.dumps(note, indent=2, sort_keys=True))


def kernel_designs(
    farm: str, sensitivity: str | None, data_dir: str, configs: str | None, out_dir: Path
) -> None:
    """the shared-kernel design of a farm's prepared rows, real and with a permuted layout."""
    from windfold.models.blocks import build_kernel_designs, kernel_basis

    protocol, _, turbine = for_farm(farm, sensitivity, folder_of(configs))
    kernel = protocol.kernel
    basis = kernel_basis(
        kernel.along_max_d,
        kernel.across_max_d,
        kernel.along_first_knot_d,
        kernel.along_knots,
        kernel.across_first_knot_d,
        kernel.across_knots,
        kernel.degree,
        len(protocol.processing.speed_bin_edges_ms) - 1,
    )
    build_kernel_designs(
        Path(data_dir) / "prepared" / prepared_folder(farm, sensitivity, protocol),
        out_dir / "prepared" / kernel_folder(farm, sensitivity, protocol),
        turbine.rotor_diameter_m,
        basis,
        kernel.along_max_d,
        kernel.across_max_d,
        protocol.seeds.study,
    )


def manifests(
    study: str,
    farm: str,
    period: str,
    sensitivity: str | None,
    data_dir: str,
    configs: str | None,
    out_dir: Path,
) -> None:
    protocol, farm_config, _ = for_farm(farm, sensitivity, folder_of(configs))
    source = prepared_folder(farm, sensitivity, protocol)
    rows = pl.read_parquet(Path(data_dir) / "prepared" / source / "rows.parquet")
    manifest, summary = splits.farm_manifest(
        rows.filter(pl.col("period") == period),
        farm_config,
        protocol.splits,
        protocol.monitoring,
        period,
        protocol.seeds.study,
    )
    target = out_dir / manifest_file(study, farm, sensitivity, protocol)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(target, compression="zstd")
    summary.write_csv(target.with_name(f"{farm}_summary.csv"))
