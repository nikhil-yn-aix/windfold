import csv
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from windfold.readers.status_logs import read_status_logs
from windfold.schemas import EVENTS, conform

ROOT = Path(__file__).resolve().parents[2]
KELMARSH_2021 = ROOT / "data/raw/kelmarsh/Kelmarsh_SCADA_2021_4456.zip"
KWF1_2021 = "Status_Kelmarsh_1_2021-01-01_-_2022-01-01_228.csv"

COLUMNS = [
    "Timestamp start",
    "Timestamp end",
    "Duration",
    "Status",
    "Code",
    "Message",
    "Comment",
    "Service contract category",
    "IEC category",
]
CONTRACTS = ["Global contract category", "Custom contract category"]
NAMES = {f"Kelmarsh {i}": f"KWF{i}" for i in range(1, 7)}
STOP_2016 = [
    "2016-01-14 19:28:03",
    "2016-01-23 14:36:32",
    "211:08:29",
    "Stop",
    "111",
    "Emergency stop nacelle",
    "",
    "Emergency stop switch (Nacelle) (11)",
    "Forced outage",
]
CURTAILMENT_2021 = [
    "2021-06-02 10:03:00",
    "2021-06-02 10:17:00",
    "00:14:00",
    "Curtailment",
    "111",
    "Grid constraint curtailment",
    "",
    "Curtailment (23)",
    "Partial Performance",
]
OPEN = ["2016-01-23 13:46:05", "-", "-", "Warning", "5100", "Service obstacle light", "", "", ""]
QUOTED = [
    "2016-02-01 08:00:00",
    "2016-02-01 08:20:00",
    "00:20:00",
    "Informational",
    "6410",
    "Manual yaw, left",
    "checked, ok",
    "Warnings (27)",
    "Full Performance",
]


def status_log(title: str, records: list[list[str]], columns: list[str] = COLUMNS) -> bytes:
    body = io.StringIO()
    writer = csv.writer(body, lineterminator="\r\n")
    writer.writerow(columns)
    writer.writerows(record + [""] * (len(columns) - len(record)) for record in records)
    preamble = [
        "# This file was exported by Greenbyte at 2022-01-27 10:52:06.",
        "#",
        f"# Turbine: {title}",
        "# Turbine type: Senvion MM92 (Senvion MM92 kW)",
        "# Time zone: UTC",
        "# Time interval: 2016-01-01 00:00:00 - 2017-01-01 00:00:00 (366 days)",
        "#",
        f"# {title} Sum production: 4239777 kWh",
        "#",
    ]
    return ("\r\n".join(preamble) + "\r\n" + body.getvalue()).encode("utf-8")


def archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in members.items():
            target.writestr(name, data)
    return path


def utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def test_status_log_records_become_events_as_published(tmp_path: Path):
    path = archive(
        tmp_path / "a.zip",
        {"Status_Kelmarsh_1_x.csv": status_log("Kelmarsh 1", [STOP_2016, OPEN, QUOTED])},
    )
    events = read_status_logs([path], NAMES)
    assert events.schema == EVENTS
    conform(events, EVENTS)
    assert events.rows(named=True) == [
        {
            "turbine": "KWF1",
            "start": utc("2016-01-14 19:28:03"),
            "end": utc("2016-01-23 14:36:32"),
            "source": "status_log",
            "status": "Stop",
            "code": 111,
            "message": "Emergency stop nacelle",
            "category": "Forced outage",
        },
        {
            "turbine": "KWF1",
            "start": utc("2016-01-23 13:46:05"),
            "end": None,
            "source": "status_log",
            "status": "Warning",
            "code": 5100,
            "message": "Service obstacle light",
            "category": None,
        },
        {
            "turbine": "KWF1",
            "start": utc("2016-02-01 08:00:00"),
            "end": utc("2016-02-01 08:20:00"),
            "source": "status_log",
            "status": "Informational",
            "code": 6410,
            "message": "Manual yaw, left",
            "category": "Full Performance",
        },
    ]


def test_a_reused_code_keeps_the_status_and_message_of_its_year(tmp_path: Path):
    paths = [
        archive(
            tmp_path / "2016.zip",
            {"Status_Kelmarsh_1_a.csv": status_log("Kelmarsh 1", [STOP_2016])},
        ),
        archive(
            tmp_path / "2021.zip",
            {
                "Status_Kelmarsh_1_b.csv": status_log(
                    "Kelmarsh 1", [CURTAILMENT_2021], COLUMNS + CONTRACTS
                )
            },
        ),
    ]
    events = read_status_logs(paths, NAMES)
    assert events.select("code", "status", "message").rows() == [
        (111, "Stop", "Emergency stop nacelle"),
        (111, "Curtailment", "Grid constraint curtailment"),
    ]


def test_identical_records_are_listed_once(tmp_path: Path):
    longer = STOP_2016[:1] + ["2016-01-23 14:36:33"] + STOP_2016[2:]
    paths = [
        archive(
            tmp_path / "2016.zip",
            {"Status_Kelmarsh_1_a.csv": status_log("Kelmarsh 1", [STOP_2016, STOP_2016, longer])},
        ),
        archive(
            tmp_path / "2017.zip",
            {"Status_Kelmarsh_1_b.csv": status_log("Kelmarsh 1", [STOP_2016], COLUMNS + CONTRACTS)},
        ),
    ]
    events = read_status_logs(paths, NAMES)
    assert events["end"].to_list() == [utc("2016-01-23 14:36:32"), utc("2016-01-23 14:36:33")]


@pytest.mark.skipif(not KELMARSH_2021.exists(), reason="kelmarsh raw files are not on this machine")
def test_a_real_turbine_year_is_read_in_full():
    with zipfile.ZipFile(KELMARSH_2021) as source:
        text = source.read(KWF1_2021).decode()
    lines = text.splitlines()
    records = list(csv.DictReader(lines[9:]))
    published = {
        (
            utc(r["Timestamp start"]),
            None if r["Timestamp end"] == "-" else utc(r["Timestamp end"]),
            r["Status"],
            int(r["Code"]),
            r["Message"] or None,
            r["IEC category"] or None,
        )
        for r in records
    }
    events = read_status_logs([KELMARSH_2021], NAMES).filter(pl.col("turbine") == "KWF1")
    read_back = events.select("start", "end", "status", "code", "message", "category").rows()
    assert len(records) > 1_000
    assert len(read_back) == len(set(read_back)) == len(published)
    assert set(read_back) == published
