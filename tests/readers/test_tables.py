import csv
import hashlib
import io
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from windfold.config import RawFile, load_farm
from windfold.readers import read_farm, write_tables
from windfold.schemas import EVENTS, LAYOUT, RAW_ROWS, conform
from windfold.sources import SourceError

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = dict(load_farm("kelmarsh").channels)
TABLES = ROOT / "data/tables"
TEN_MINUTES = timedelta(minutes=10)


def greenbyte_csv(title: str, header: list[str], records: list[list[str]], data: bool) -> bytes:
    body = io.StringIO()
    writer = csv.writer(body, lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(records)
    head, rest = body.getvalue().split("\r\n", 1)
    preamble = f"# Exported by Greenbyte.\r\n#\r\n# Turbine: {title}\r\n# Time zone: UTC\r\n#\r\n"
    return (preamble + ("# " if data else "") + head + "\r\n" + rest).encode()


@pytest.fixture
def raw_farm(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    columns = list(CHANNELS.values())
    values = ["905.25", "6.9", "0.8", "212.5", "210.0", "11.5", "4.2", "2060", "1990", "1", "0"]
    data = greenbyte_csv(
        "Kelmarsh 1",
        ["Date and time", *columns],
        [["2016-01-03 00:10:00", *values], ["2016-01-03 00:00:00", *values]],
        data=True,
    )
    status = greenbyte_csv(
        "Kelmarsh 1",
        [
            "Timestamp start",
            "Timestamp end",
            "Duration",
            "Status",
            "Code",
            "Message",
            "IEC category",
        ],
        [
            [
                "2016-01-03 00:03:00",
                "2016-01-03 00:17:00",
                "00:14:00",
                "Stop",
                "111",
                "Emergency stop nacelle",
                "Forced outage",
            ]
        ],
        data=False,
    )
    with zipfile.ZipFile(raw_dir / "Kelmarsh_SCADA_2016_3082.zip", "w") as target:
        target.writestr("Turbine_Data_Kelmarsh_1_2016.csv", data)
        target.writestr("Status_Kelmarsh_1_2016.csv", status)
    (raw_dir / "Kelmarsh_WT_static.csv").write_text(
        "Title,Alternative Title,Hub Height (m),Latitude,Longitude\r\n"
        "Kelmarsh 1,KWF1,78.5,52.400604,-0.947133\r\n"
    )
    files = tuple(
        RawFile(name=p.name, bytes=p.stat().st_size, md5=hashlib.md5(p.read_bytes()).hexdigest())
        for p in sorted(raw_dir.iterdir())
    )
    return load_farm("kelmarsh").model_copy(update={"raw_files": files}), raw_dir


def test_write_tables_writes_rows_and_events_as_zstd_parquet(raw_farm, tmp_path: Path):
    farm, raw_dir = raw_farm
    out_dir = tmp_path / "tables" / "kelmarsh"
    write_tables(farm, raw_dir, out_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "events.parquet",
        "layout.parquet",
        "rows.parquet",
    ]
    layout = conform(pl.read_parquet(out_dir / "layout.parquet"), LAYOUT)
    assert layout.select("turbine", "x_m", "y_m", "hub_height_m").row(0) == ("KWF1", 0.0, 0.0, 78.5)
    rows, events = read_farm(farm, raw_dir)
    for name, frame, schema in (("rows", rows, RAW_ROWS), ("events", events, EVENTS)):
        written = pl.read_parquet(out_dir / f"{name}.parquet")
        assert written.equals(frame)
        conform(written, schema, extra=True)
        metadata = pq.ParquetFile(out_dir / f"{name}.parquet").metadata
        assert metadata.row_group(0).column(0).compression == "ZSTD"
    assert rows.select("turbine", "timestamp").rows() == [
        ("KWF1", datetime(2016, 1, 3, 0, 0, tzinfo=UTC)),
        ("KWF1", datetime(2016, 1, 3, 0, 10, tzinfo=UTC)),
    ]
    assert events.select("turbine", "code", "message").rows() == [
        ("KWF1", 111, "Emergency stop nacelle")
    ]


def test_read_farm_stops_with_the_fix_when_raw_files_are_missing(tmp_path: Path):
    with pytest.raises(
        SourceError, match="kelmarsh has no raw files yet. run windfold fetch kelmarsh"
    ):
        read_farm(load_farm("kelmarsh"), tmp_path)


def time_grid(farm: str, minutes: int = 10) -> pl.DataFrame:
    path = TABLES / farm / "rows.parquet"
    if not path.exists():
        pytest.skip(f"no {farm} raw table on this machine")
    rows = pl.read_parquet(path, columns=["turbine", "timestamp"])
    assert rows.select(pl.struct("turbine", "timestamp").is_duplicated().any()).item() is False
    step = timedelta(minutes=minutes)
    assert rows.filter(pl.col("timestamp").dt.truncate(step) != pl.col("timestamp")).is_empty()
    return (
        rows.group_by("turbine")
        .agg(
            rows=pl.len(),
            first=pl.col("timestamp").min(),
            last=pl.col("timestamp").max(),
            gaps=pl.col("timestamp")
            .sort()
            .diff()
            .filter(pl.col("timestamp").sort().diff() != step),
        )
        .sort("turbine")
    )


def test_kelmarsh_rows_match_the_data_card():
    grid = time_grid("kelmarsh")
    assert grid["turbine"].to_list() == [f"KWF{i}" for i in range(1, 7)]
    assert grid["rows"].to_list() == [473_184] * 6
    assert grid["rows"].sum() == 2_839_104
    assert set(grid["first"]) == {datetime(2016, 1, 3, tzinfo=UTC)}
    assert set(grid["last"]) == {datetime(2024, 12, 31, 23, 50, tzinfo=UTC)}
    assert all(len(gaps) == 0 for gaps in grid["gaps"])


def test_penmanshiel_rows_match_the_data_card_without_its_2024_file():
    grid = time_grid("penmanshiel")
    starts = {
        "T01": (6, 6),
        "T02": (6, 3),
        "T04": (6, 13),
        "T05": (6, 15),
        "T06": (6, 2),
        "T07": (6, 2),
        "T08": (7, 27),
        "T09": (6, 24),
        "T10": (6, 27),
        "T11": (7, 19),
        "T12": (7, 2),
        "T13": (6, 30),
        "T14": (7, 9),
        "T15": (7, 14),
    }
    assert grid["turbine"].to_list() == list(starts)
    assert grid["rows"].sum() == 5_758_976 - 5 * 52_704
    assert [(t.month, t.day) for t in grid["first"]] == list(starts.values())
    assert set(grid["last"]) == {datetime(2023, 12, 31, 23, 50, tzinfo=UTC)}
    january_2023 = [timedelta(days=31, minutes=10)]
    for turbine, gaps in zip(grid["turbine"], grid["gaps"], strict=True):
        assert gaps.to_list() == (january_2023 if turbine <= "T10" else []), turbine


def test_smarteole_rows_match_the_data_card():
    grid = time_grid("smarteole", minutes=1)
    assert grid["turbine"].to_list() == [f"SMV{i}" for i in range(1, 8)]
    assert grid["rows"].to_list() == [134_661] * 7
    assert set(grid["first"]) == {datetime(2020, 2, 17, 16, 30, tzinfo=UTC)}
    assert set(grid["last"]) == {datetime(2020, 5, 24, 23, 59, tzinfo=UTC)}


def test_hill_of_towie_rows_line_up_with_the_start_labelled_shutdown_log():
    rows_path = TABLES / "hill_of_towie/rows.parquet"
    log_path = ROOT / "data/raw/hill_of_towie/Hill_of_Towie_ShutdownDuration.zip"
    if not (rows_path.exists() and log_path.exists()):
        pytest.skip("no hill of towie raw table or shutdown log on this machine")
    rows = pl.read_parquet(rows_path, columns=["turbine", "timestamp", "running_s"])
    with zipfile.ZipFile(log_path) as archive:
        log = pl.read_csv(io.BytesIO(archive.read("ShutdownDuration.csv")), infer_schema=False)
    stopped = log.filter(pl.col("ShutdownDuration") == "600").select(
        pl.col("TurbineName").alias("turbine"),
        pl.col("TimeStamp_StartFormat").str.to_datetime("%Y-%m-%d %H:%M:%S%z").alias("timestamp"),
    )

    def idle_share(shift: timedelta) -> float:
        joined = stopped.with_columns(pl.col("timestamp") + shift).join(
            rows, on=["turbine", "timestamp"]
        )
        return joined.select((pl.col("running_s") == 0).mean()).item()

    aligned = idle_share(timedelta(0))
    assert aligned > 0.999
    assert aligned > idle_share(TEN_MINUTES) and aligned > idle_share(-TEN_MINUTES)
