"""step 5: power-curve bounds against a published or fitted curve."""

import numpy as np
import polars as pl

from windfold.config import Curve


def curve_power(speed: np.ndarray, curve: Curve) -> np.ndarray:
    """the curve's power at each speed: linear between table points, zero below the first and
    flat above the last.
    """
    return np.interp(speed, np.asarray(curve.speed_ms), np.asarray(curve.value), left=0.0)


def fit_power_curve(
    speed: np.ndarray, power: np.ndarray, bin_width: float, min_rows: int, source: str
) -> Curve:
    """median power in speed bins of bin_width, kept where a bin holds at least min_rows rows.

    each bin's median is placed at the bin center.
    """
    usable = np.isfinite(speed) & np.isfinite(power)
    frame = pl.DataFrame({"speed": speed[usable], "power": power[usable]}).with_columns(
        (pl.col("speed") / bin_width).floor().alias("bin")
    )
    table = (
        frame.group_by("bin")
        .agg(pl.col("power").median(), pl.len().alias("rows"))
        .filter(pl.col("rows") >= min_rows)
        .sort("bin")
    )
    centres = ((table.get_column("bin") + 0.5) * bin_width).to_list()
    return Curve(
        speed_ms=tuple(centres), value=tuple(table.get_column("power").to_list()), source=source
    )


def outside_curve_bounds(
    speed: np.ndarray,
    power: np.ndarray,
    curve: Curve,
    rated_kw: float,
    speed_tolerance: float,
    power_tolerance: float,
) -> np.ndarray:
    """true where power lies outside the curve shifted by speed_tolerance and power_tolerance.

    the upper bound is the curve at speed times (1 + speed_tolerance) plus power_tolerance of
    rated, the lower bound the curve at speed times (1 - speed_tolerance) minus the same margin.
    rows without speed or power are never flagged here.
    """
    upper = curve_power(speed * (1.0 + speed_tolerance), curve) + power_tolerance * rated_kw
    lower = curve_power(speed * (1.0 - speed_tolerance), curve) - power_tolerance * rated_kw
    with np.errstate(invalid="ignore"):
        return (power > upper) | (power < lower)
