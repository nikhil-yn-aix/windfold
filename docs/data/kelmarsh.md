# Kelmarsh

Cubico's Kelmarsh wind farm in West Northamptonshire, England: six Senvion MM92 turbines, KWF1 to KWF6, rated 2050 kW. hub height is 78.5 m for KWF1, KWF2, KWF4 and KWF5 and 68.5 m for KWF3 and KWF6. the rotor is 92.5 m (manufacturer brochure; the static file rounds it to 92 m). commercial operation started on 2016-04-15.

- source: Zenodo record "Kelmarsh wind farm data", DOI 10.5281/zenodo.16807551 (concept DOI 10.5281/zenodo.5841833), Plumley and Takeuchi 2025, Cubico.
- license: CC BY 4.0. the record asks users to tell Cubico how the data is used.
- raw files: the SCADA zips 2016 to 2024, the static and signal-mapping files and the KMZ layout, each listed with bytes and MD5 in `src/windfold/configs/farms/kelmarsh.toml`.

## what we read

the 10-minute Greenbyte export, one row per turbine and interval start in UTC: 2,839,104 rows, 473,184 per turbine, from 2016-01-03 00:00 to 2024-12-31 23:50.

| column | Greenbyte signal | present from |
|---|---|---|
| power_kw | Power (kW) | 2016-01-21 |
| speed | Wind speed Sensor 2 (m/s) | 2017-06-12 09:00 |
| speed_std | Wind speed Sensor 2, Standard deviation (m/s) | 2017-09-25 |
| wind_direction | Wind direction (°), nacelle position plus vane | 2016-05-03 |
| nacelle_position | Nacelle position (°) | 2016-05-03 |
| ambient_temperature | Nacelle ambient temperature (°C) | 2016-05-03 |
| power_setpoint_kw | Turbine Power setpoint (kW), the interval mean | 2018-02-07 |
| power_setpoint_min_kw | Turbine Power setpoint, Min (kW) | 2020-01-26 |
| time_based_availability | Time-based System Avail. | 2016-01-14 |
| downtime_loss_kwh | Lost Production to Downtime (kWh) | 2016-01-21 |

the per-turbine status logs become 503,471 events (start, end, status, code, message, IEC category), all in UTC. the statuses are Informational (487,605), Stop (7,045), Warning (6,491), Curtailment (1,446) and Communication (884).

the layout is the readers' `data/tables/kelmarsh/layout.parquet`, copied to `data/prepared/kelmarsh/layout.parquet`. its east and north meters reproduce the WGS84 geodesic distance of every turbine pair to the millimeter. the rotor diameter (92.5 m) comes from the turbine config.

## quirks

- the primary "Wind speed" channel changes meaning: the mean of sensors 1 and 2 until late 2017, a day-by-day mix from December 2017 to April 2018, then sensor 2 alone, a step of about 0.2 m/s. we read sensor 2 everywhere. it first appears at 2017-06-12 09:00 UTC on all six turbines, so the calibration window is 2017-06-12 to 2017-12-31 and the development window 2018 to 2024.
- sensor 2's standard deviation and minimum start on 2017-09-25. before that no row has a speed spread, so turbulence intensity is missing in 50,613 of the 106,764 prepared calibration rows, and in none after.
- the 2023 and 2024 exports are paginated: each timestamp appears many times and only the first copy holds the measurements. KWF1 at 2023-01-05 13:00 has 81 copies, and the first holds 569.303 kW. the reader keeps the first copy of each timestamp in file order.
- sensor 2 reads exactly −15.000 in 1,973 intervals, mostly KWF4 in 2023 (wind-up's finding). in 2,686 intervals the interval minimum is negative. the reader makes −15.000, negative speeds and both of these cases missing, speed and spread together.
- the setpoint is a controller reference, not a cap: below rated it sits far above the power (median 1,089 kW against 242 kW at sensor-2 speeds of 4.5 to 5.5 m/s, and never under 412 kW). only its interval minimum, or the mean before 2020, can show a cap.
- status codes are reused across years: 111 is "Emergency stop nacelle" (Stop) in 2016 and "Grid constraint curtailment" (Curtailment) from 2021. events are matched on status plus message.
- the record is empty for every turbine from 2018-10-24 to 2018-11-01.
- direction: KWF6's north is about 50° off from 2023-06-15 to 2024-07-06. KWF1 steps by about 10° on 2021-12-19. KWF3 sits about +3° and KWF5 about −6° from the other turbines, and KWF4 drifts to about −5° in 2023 and 2024.
- wind-up's corrections for KWF5 change on 2020-04-16 and 2020-04-23. those corrections are anchored to ERA5 and applied to the nacelle position. against the other turbines, KWF5 shows no step that week: −6.0° before, −6.8° during and −6.5° after.
- the free-stream turbines do not see the same wind. on the calibration window, at reference speeds of 5 to 9 m/s, each turbine's speed over the median of the other free-stream turbines ranges from 0.940 (KWF6) to 1.026 (KWF2, KWF4). its power over theirs moves with it, as the power curve predicts from the speeds: KWF6 makes 0.817 of their power where the curve says 0.826, and KWF2 1.102 where it says 1.081. a miscalibrated anemometer would leave the power ratio near 1, so most of the difference is wind, not sensor. that is why the anemometer factors below reach only the reference speed: each turbine keeps its measured speed for its own checks.

## what we drop and why

every dropped row is counted once, under the first rule that drops it (`data/prepared/kelmarsh/ledger.csv`).

| step | rule | rows dropped | rows kept | timestamps kept |
|---|---|---|---|---|
| 1 | read | 0 | 2,839,104 | 473,184 |
| 2 | windows | 454,464 | 2,384,640 | 397,440 |
| 3 | power_missing | 23,202 | 2,361,438 | 394,450 |
| 3 | availability | 63,687 | 2,297,751 | 393,450 |
| 3 | downtime | 4 | 2,297,747 | 393,450 |
| 3 | stop_event | 0 | 2,297,747 | 393,450 |
| 3 | curtailment_event | 7,179 | 2,290,568 | 393,116 |
| 3 | derate_event | 15,349 | 2,275,219 | 393,116 |
| 3 | unknown_event | 22,907 | 2,252,312 | 389,664 |
| 3 | binding_setpoint | 9,955 | 2,242,357 | 389,626 |
| 3 | power_floor | 289,461 | 1,952,896 | 353,857 |
| 4 | speed_missing | 345 | 1,952,551 | 353,803 |
| 4 | direction_missing, out_of_range, frozen_row, stuck_speed, stuck_direction | 0 | 1,952,551 | 353,803 |
| 5 | power_curve | 4,656 | 1,947,895 | 353,780 |
| 7 | manual_yaw | 0 | 1,947,895 | 353,780 |
| 7 | direction_jump | 263 | 1,947,632 | 353,780 |
| 7 | offset_unknown | 813 | 1,946,819 | 353,706 |
| 8 | icing_event | 0 | 1,946,819 | 353,706 |
| 9 | wake_source | 65,080 | 1,881,739 | 350,860 |
| 10 | reference | 180 | 1,881,559 | 350,726 |
| 10 | speed_window | 606,273 | 1,275,286 | 227,973 |
| 11 | valid_turbines | 975 | 1,274,311 | 226,998 |

- windows: rows before 2017-06-12, when sensor 2 starts.
- status: an interval is dropped if availability is below 1, downtime loss is above 0 or the power is unknown. it is also dropped if it overlaps a logged stop, curtailment, power reduction or communication gap, if a setpoint below 98% of rated holds the power (power at or above the lowest known setpoint minus 5% of rated) or if the power is below 1% of rated. every logged stop already has availability below 1, so stop_event drops nothing new.
- the curtailments are "Technical curtailment" (2022 and 2023, caps of 1000 and 1666 kW) and "Grid constraint curtailment" (2023, cap 412 kW). the setpoint rule also catches intervals whose cap changed within them.
- power curve: bounds of ±10% in speed and ±5% of rated in power around the curve fitted on the calibration window, both on measured speeds. 4,656 rows (0.24%) fall outside. the published MM92 curve dropped 11,450, and 96% of those were good rows above it. applying the anemometer factors to each turbine's own speed would have added 2,045 drops, nearly all on KWF2, KWF4, KWF5 and KWF6, whose power follows their measured speed.
- direction: the offset detector drops the rows of the day around each jump it finds (KWF1 2021-12-19 to 20, KWF6 2023-06-15 to 16 and 2024-07-05 to 06). rows with too little history for an offset are dropped too.
- icing: only logged icing events count. none reaches this step, because every logged icing stop already has availability below 1.
- wake sources: a row is kept only if every turbine within 30 diameters whose disturbed sector holds the wind is operating normally.
- inputs: the reference speed is the median of the free-stream turbines' nacelle speeds, each times its anemometer factor. it must lie between 5 and 10 m/s.
- the ledger also counts waked rows stopped below the power floor: 2,122 at reference speeds 5 to 6.5 m/s, 47 at 6.5 to 8 and 0 at 8 to 10. log power cannot represent a stopped turbine, so near 5 m/s the prepared rows miss some deep wakes.

prepared rows (`data/prepared/kelmarsh/rows.parquet`):

| period | rows | timestamps | waked share | valid turbines per timestamp |
|---|---|---|---|---|
| calibration (2017-06-12 to 2017-12-31) | 106,764 | 18,195 | 0.487 | 5.87 |
| development (2018 to 2024) | 1,167,547 | 208,803 | 0.463 | 5.59 |

the speed bins hold 519,424, 440,359 and 314,528 rows (5 to 6.5, 6.5 to 8 and 8 to 10 m/s), waked in 47.1%, 46.4% and 45.4% of them. 1.6% of timestamps (3,700) have exactly one free-stream turbine, so their reference speed is that turbine's speed times its factor. 96% of timestamps are steady. the target sums to zero at every timestamp (largest |Σy| 6e-15).

## calibration from the data

- anemometer factors, fitted on the calibration window with at least 1,000 qualifying timestamps per turbine: for each turbine, the median over timestamps where it and at least one other turbine are free-stream of the other free-stream turbines' median speed over its own. the factors reach the reference speed only, the median of the free-stream speeds each times its factor; the power-curve bounds, the dip fit and the speed column keep measured speeds (recorded in `provenance.json`).

  | turbine | KWF1 | KWF2 | KWF3 | KWF4 | KWF5 | KWF6 |
  |---|---|---|---|---|---|---|
  | factor | 1.0257 | 0.9746 | 1.0160 | 0.9760 | 0.9900 | 1.0622 |
  | timestamps | 12,857 | 16,565 | 5,758 | 11,574 | 2,621 | 5,106 |

  on the prepared rows the reference speed differs from the median of the measured free-stream speeds by a median of +0.001 m/s (5 to 95%: −0.195 to +0.240, at most 0.585), and 84,584 rows (6.6%) sit in another speed bin than they would without factors. the factors put the free-stream turbines on one basis, whichever of them are free-stream.
- farm rotation: +0.83°, the median wake-dip offset of 18 turbine pairs 2 to 10 diameters apart on the calibration window. the dips sit on the layout bearings after it (`runs/inspect/kelmarsh/wake_dips.pdf`).
- direction offsets: each turbine's median deviation from the other turbines over the previous 30 days, in two passes. the first pass measures each turbine against the median of the others as read. while KWF6 sat 50° off, it moved that median by one rank for every other turbine, and KWF4 was left about +2° off in 2023 and 2024. the second pass measures each turbine against the median of the others less their first-pass offsets. after it, every turbine's yearly median deviation from the others lies within 0.22° (mean 0.08°), and KWF4's quarterly medians from mid-2023 lie within 0.3° of zero. a jump is declared when the 24-hour median differs from the offset by more than 10°, and the 30 days then restart (`runs/inspect/kelmarsh/direction_offsets.pdf`).
- power curve: fitted on the calibration window (bin medians every 0.5 m/s), on measured speeds.
- temperature: readings outside −40 to 50 °C would be made missing; none is.

## checks by hand

- KWF1 at 2020-01-10 05:00: the raw CSV line in `Turbine_Data_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv` holds 633.418 kW and 6.96822 m/s, as the readers' row and the prepared row do. the row is waked. the reference speed of 7.1611 m/s is the median of the two free-stream turbines after their factors, KWF2 (7.5644 × 0.974635 = 7.3725) and KWF3 (6.8403 × 1.015988 = 6.9496), and its y of 0.258332 equals log power minus the mean log power of the six turbines.
- KWF6 at 2022-09-26 02:30 (717 kW under a 1666 kW cap) is dropped by the logged curtailment. the setpoint rule alone would keep it, because its power sits far below the cap.

## histories

every monitoring history from 2018-01-01, 04-01, 07-01 and 10-01 keeps at least 300 training timestamps. the 30-day threshold fit from 2018-10-01 falls short: its threshold days fall in the empty record of 2018-10-24 to 11-01 (`runs/development/manifests/kelmarsh_summary.csv`).
