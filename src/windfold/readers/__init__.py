"""raw tables: each farm's provider files read into a row table, an event table and a layout."""

import fnmatch
import os
from datetime import timedelta
from pathlib import Path

import polars as pl

from windfold import sources
from windfold.config import FarmConfig
from windfold.readers import greenbyte, hill_of_towie, layout, smarteole, status_logs


def archives_of(farm: FarmConfig, raw_dir: Path) -> list[Path]:
    pattern = farm.source.archives
    archives = [
        raw_dir / raw.name for raw in farm.raw_files if fnmatch.fnmatchcase(raw.name, pattern)
    ]
    if not archives:
        raise ValueError(f"no raw file of {farm.name} matches {pattern}")
    return archives


def read_farm(farm: FarmConfig, raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """the raw rows and events of farm, read from its raw files in raw_dir once they verify."""
    sources.verify(farm, raw_dir)
    archives = archives_of(farm, raw_dir)
    interval = timedelta(minutes=farm.rules.interval_minutes)
    match farm.reader:
        case "greenbyte":
            names = greenbyte.read_turbine_names(raw_dir / farm.source.turbines)
            rows = greenbyte.read_rows(
                archives,
                names,
                farm.channels,
                farm.source.null_values,
                farm.source.speed_missing_codes,
            )
            return rows, status_logs.read_status_logs(archives, names)
        case "hill_of_towie":
            stations = hill_of_towie.read_stations(raw_dir / farm.source.turbines)
            rows = hill_of_towie.read_rows(archives, stations, farm.channels, interval)
            shutdown_logs = [raw_dir / name for name in farm.source.event_files]
            events = pl.concat(
                [
                    hill_of_towie.read_alarms(archives, stations),
                    hill_of_towie.read_shutdowns(shutdown_logs, stations.values(), interval),
                ]
            )
            return rows, events.sort("turbine", "start", maintain_order=True)
        case "smarteole":
            if len(archives) != 1:
                raise ValueError(f"smarteole reads one package, not {[p.name for p in archives]}")
            turbines = layout.read_layout(farm, raw_dir).get_column("turbine").to_list()
            rows = smarteole.read_rows(archives[0], turbines, farm.channels)
            return rows, smarteole.read_control_log(archives[0], interval)


def write_tables(farm: FarmConfig, raw_dir: Path, out_dir: Path) -> None:
    """write the rows, events and layout of farm to out_dir as rows, events and layout.parquet."""
    rows, events = read_farm(farm, raw_dir)
    turbines = layout.read_layout(farm, raw_dir)
    strangers = set(rows.get_column("turbine").unique()) - set(turbines.get_column("turbine"))
    if strangers:
        raise ValueError(f"{farm.name} rows have turbines {sorted(strangers)} the layout lacks")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("rows", rows), ("events", events), ("layout", turbines)):
        partial = out_dir / f"{name}.parquet.partial"
        frame.write_parquet(partial, compression="zstd")
        os.replace(partial, out_dir / f"{name}.parquet")
