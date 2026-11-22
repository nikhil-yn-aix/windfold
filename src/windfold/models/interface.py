"""what every model gives the runner: fit on training rows, predict the centered contrast."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl


@dataclass(frozen=True)
class FitContext:
    """where a fit reads its farm's prepared files and what it may use.

    kernel_dir holds the shared-kernel design; None means prepared_dir / "kernel_design".
    """

    farm: str
    prepared_dir: Path
    seed: int
    threads: int
    device: str | None
    kernel_dir: Path | None = None

    @property
    def kernel_design(self) -> Path:
        return self.kernel_dir or self.prepared_dir / "kernel_design"


class Fitted(Protocol):
    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        """one output row per input row, in input order, with Float64 columns mean and variance.

        both are on the centered log-power scale of the target y. variance is null for models
        without a predictive distribution. a model whose predictions rest on a count of training
        rows, such as the direction lookup, may add an Int64 column support with that count. a
        model whose variance is costly may also offer predict_mean(rows), the same means as a
        float array, which the training residuals use.
        """
        ...


class Model(Protocol):
    name: str

    def fit(self, train: pl.DataFrame, context: FitContext) -> Fitted:
        """fit on the prepared rows of the training timestamps, each timestamp's rows together."""
        ...


def check_whole_timestamps(rows: pl.DataFrame) -> None:
    """every timestamp brings exactly one row for each of its valid turbines."""
    per_time = rows.group_by("timestamp").agg(
        pl.len().alias("rows"),
        pl.col("turbine").n_unique().alias("turbines"),
        pl.col("valid_turbines").min().alias("fewest"),
        pl.col("valid_turbines").max().alias("most"),
    )
    broken = per_time.filter(
        (pl.col("rows") != pl.col("turbines"))
        | (pl.col("rows") != pl.col("fewest"))
        | (pl.col("fewest") != pl.col("most"))
    )
    if broken.height:
        first = broken.row(0, named=True)
        raise ValueError(
            f"{broken.height} timestamps do not hold one row for each of their valid_turbines; "
            f"{first['timestamp']} has {first['rows']} rows and valid_turbines {first['fewest']}"
        )
