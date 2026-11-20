# Penmanshiel

Cubico's Penmanshiel wind farm in the Scottish Borders, Scotland: fourteen Senvion MM82 turbines named T01 to T15, with no T03. they have an 82 m rotor, a 59 m hub and are rated 2050 kW. commercial operation started on 2016-09-01.

- source: Zenodo record "Penmanshiel wind farm data", DOI 10.5281/zenodo.16807304 (concept DOI 10.5281/zenodo.5946807), Plumley and Takeuchi 2025, Cubico.
- license: CC BY 4.0. the record asks users to tell Cubico how the data is used.
- raw files: the SCADA zips 2016 to 2023 for both turbine groups (2023 in part files), the static and signal-mapping files and the KMZ layout, each listed with bytes and MD5 in `src/windfold/configs/farms/penmanshiel.toml`. the 2024 zip holds only T11 to T15 and is not used.

## what we read

the 10-minute Greenbyte export, one row per turbine and interval start in UTC: 5,495,456 rows from 2016-06-02 17:40 to 2023-12-31 23:50.

| column | Greenbyte signal | present from |
|---|---|---|
| power_kw | Power (kW) | 2016-06-02 |
| speed | Wind speed Sensor 2 (m/s) | 2016-06-02 |
| speed_std | Wind speed Sensor 2, Standard deviation (m/s) | 2016-06-02 |
| wind_direction | Wind direction (°), nacelle position plus vane | 2016-06-02 |
| nacelle_position | Nacelle position (°) | 2016-06-02 |
| ambient_temperature | Nacelle ambient temperature (°C) | 2016-06-02 |
| power_setpoint_kw | Turbine Power setpoint (kW), the interval mean | 2018-04-05 |
| power_setpoint_min_kw | Turbine Power setpoint, Min (kW) | 2020-01-26 |
| time_based_availability | Time-based System Avail. | 2016-06-02 |
| downtime_loss_kwh | Lost Production to Downtime (kWh) | 2016-06-02 |

the per-turbine status logs become 999,881 events in UTC. the statuses are Informational (966,820), Stop (20,021), Warning (11,742), Communication (1,191) and Curtailment (107).

the layout is the readers' `data/tables/penmanshiel/layout.parquet`, copied to `data/prepared/penmanshiel/layout.parquet`. its east and north meters reproduce the WGS84 geodesic distance of every turbine pair to the millimeter. the rotor diameter (82 m) comes from the turbine config.

## quirks

- the primary "Wind speed" channel changes meaning, as at Kelmarsh; we read sensor 2, present from the first rows. the step between the two definitions is only about 0.05 m/s here.
- the 2023 exports are paginated, and the reader keeps the first copy of each timestamp in file order; the first copy carries the measurements.
- in 6,230 intervals the speed's interval minimum is negative, and the reader makes the speed and its spread missing: 2,458 on T06 in 2016, 1,003 on T02 in 2016, 656 on T13 in 2021 and 551 on T10 in 2023, the rest scattered.
- the turbines start on different dates during commissioning, between 2016-06-02 (T07) and 2016-07-27 (T08), before the 2016-09-01 commercial operation date. January 2023 is absent for T01 to T10 (there is no T03); only T11 to T15 report that month. these are gaps in the published files, not losses in the reader.
- "P output externally reduced" covers much of the summer and autumn of 2016: 3,639 turbine-hours in July, 5,703 in August, 6,523 in September and 1,639 in November. there is no setpoint before 2018, so the status log is the only defense there.
- direction: the whole farm sits about 7° counterclockwise of the layout bearings in every year. the turbines agree with each other within a few degrees, so this is a common north offset.
- lone 10-minute rows can sit 20 to 60° from the other turbines; the jump detector needs 72 rows in its 24-hour window, so it leaves lone rows alone. with 14 turbines they barely move the farm direction.
- the free-stream turbines do not see the same wind. on the calibration window, at reference speeds of 5 to 9 m/s, T07's speed is 0.905 of the other free-stream turbines' median and its power 0.693 of theirs, where the power curve predicts 0.723 from the speeds. T12 reads 1.042 and makes 1.233 (predicted 1.140), T13 1.041 and 1.138 (1.139), T15 1.032 and 1.194 (1.104). a miscalibrated anemometer would leave the power ratio near 1, so these are mostly differences in the wind (the farm sits on hilly ground), and the anemometer factors below reach only the reference speed.
- the published MM82 curve (the guaranteed curve shipped with the SMARTEOLE record) matches the measured power within a few percent from 6 m/s. it sits above it near cut-in (measured power is about 85% of the curve at 4 to 5 m/s).

## what we drop and why

every dropped row is counted once, under the first rule that drops it (`data/prepared/penmanshiel/ledger.csv`).

| step | rule | rows dropped | rows kept | timestamps kept |
|---|---|---|---|---|
| 1 | read | 0 | 5,495,456 | 398,630 |
| 2 | windows | 136,784 | 5,358,672 | 385,632 |
| 3 | power_missing | 73,447 | 5,285,225 | 382,228 |
| 3 | availability | 133,167 | 5,152,058 | 380,881 |
| 3 | downtime | 20 | 5,152,038 | 380,881 |
| 3 | stop_event | 1 | 5,152,037 | 380,881 |
| 3 | curtailment_event | 184 | 5,151,853 | 380,866 |
| 3 | derate_event | 80,881 | 5,070,972 | 376,546 |
| 3 | unknown_event | 69,982 | 5,000,990 | 371,817 |
| 3 | binding_setpoint | 71,544 | 4,929,446 | 370,674 |
| 3 | power_floor | 923,882 | 4,005,564 | 326,801 |
| 4 | speed_missing | 56 | 4,005,508 | 326,801 |
| 4 | stuck_speed | 188 | 4,005,320 | 326,799 |
| 4 | stuck_direction | 3 | 4,005,317 | 326,799 |
| 4 | direction_missing, out_of_range, frozen_row | 0 | 4,005,317 | 326,799 |
| 5 | power_curve | 16,205 | 3,989,112 | 326,645 |
| 7 | manual_yaw | 267 | 3,988,845 | 326,645 |
| 7 | direction_jump | 1,041 | 3,987,804 | 326,641 |
| 7 | offset_unknown | 4,144 | 3,983,660 | 325,238 |
| 8 | icing_event | 0 | 3,983,660 | 325,238 |
| 9 | wake_source | 299,569 | 3,684,091 | 317,572 |
| 10 | reference | 523 | 3,683,568 | 317,246 |
| 10 | speed_window | 1,429,823 | 2,253,745 | 175,718 |
| 11 | valid_turbines | 628 | 2,253,117 | 175,090 |

- windows: rows before 2016-09-01, the calibration start.
- status: the same rules as Kelmarsh. the power reductions above make derate_event the largest event rule (80,881 rows). from 2018, the setpoint rule adds 71,544.
- power curve: 16,205 rows (0.40%) fall outside the bounds around the published curve, on measured speeds, almost all below the lower bound: underperformance, as the bounds intend. applying the anemometer factors to each turbine's own speed would have added 18,162 drops, 13,155 of them on T07, whose factor of 1.104 would lift its speed above what its power shows.
- direction: 13 jumps found and the day around each dropped: T02 (2021-03-04), T08 (2018-06-12, 2021-04-23), T12 (2016-10-07), T15 (2023-05-06) and eight brief excursions on T11 between 2017 and 2023. T12's and T11's 2022-10-21 excursions are borderline: their 24-hour medians reach 10 to 10.6° from the offset, just over the threshold.
- icing: only logged icing events count, and none reaches this step.
- the nacelle temperature is missing in 1.4% of the prepared rows; no rule needs it.
- wake sources and inputs: as at Kelmarsh. with 14 turbines in rows, a stopped neighbor invalidates more rows (299,569).
- waked rows stopped below the power floor: 11,685 at reference speeds 5 to 6.5 m/s, 458 at 6.5 to 8 and 23 at 8 to 10.

prepared rows (`data/prepared/penmanshiel/rows.parquet`):

| period | rows | timestamps | waked share | valid turbines per timestamp |
|---|---|---|---|---|
| calibration (2016-09-01 to 2017-02-28) | 129,312 | 10,216 | 0.639 | 12.66 |
| development (2017-03-01 to 2019-12-31) | 953,852 | 71,242 | 0.646 | 13.39 |
| confirmation period (2020 to 2023) | 1,169,953 | 93,632 | 0.632 | 12.50 |

the speed bins hold 846,148, 735,092 and 671,877 rows (5 to 6.5, 6.5 to 8 and 8 to 10 m/s), waked in 63.1%, 64.3% and 64.2% of them. 0.44% of timestamps (775) have exactly one free-stream turbine. 92% of timestamps are steady. the target sums to zero at every timestamp (largest |Σy| 1.3e-14).

## calibration from the data

- anemometer factors, fitted on the calibration window as at Kelmarsh, with at least 1,000 qualifying timestamps per turbine; T09 (24) and T10 (808) are almost always waked there and keep 1. as at Kelmarsh, they reach the reference speed only.

  | turbine | T01 | T02 | T04 | T05 | T06 | T07 | T08 | T11 | T12 | T13 | T14 | T15 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | factor | 0.964 | 1.000 | 1.011 | 1.005 | 0.994 | 1.104 | 1.017 | 0.996 | 0.957 | 0.960 | 0.983 | 0.970 |

  T02's factor is exactly 1: speeds are logged to 0.01 m/s, so the ratio is exactly 1 at 0.6% of its timestamps and the median lands there. on the prepared rows the reference speed differs from the median of the measured free-stream speeds by a median of −0.011 m/s (5 to 95%: −0.245 to +0.292, at most 0.94), and 158,561 rows (7.0%) sit in another speed bin.
- farm rotation: +6.97°, the median wake-dip offset of 63 turbine pairs 2 to 10 diameters apart on the calibration window. before it, the median dip sits about 7° counterclockwise of the bearings; after it, on them (`runs/inspect/penmanshiel/wake_dips.pdf`).
- direction offsets and the jump detector: two passes, as at Kelmarsh. no turbine here sits far off north for long, so the second pass changes little: the mean yearly deviation from the others falls from 0.105° to 0.059° on the rows reaching step 7 (`runs/inspect/penmanshiel/direction_offsets.pdf`).
- power curve: the published MM82 curve from the turbine config, with the 3.5 m/s cut-in from the manufacturer brochure.
- temperature: readings outside −40 to 50 °C would be made missing; none is.

## checks by hand

- T04 at 2019-01-01 00:30: the readers' row equals the raw CSV line in the zip, and the prepared row keeps its 9.7989 m/s. it is kept and free-stream. the reference speed of 9.580 m/s is the median of the four free-stream turbines after their factors (T01 10.4794 × 0.964235 = 10.1046, T02 9.2485, T04 9.7989 × 1.011494 = 9.9115, T08 8.0717 × 1.01676 = 8.2070), and its y of 0.294797 equals log power minus the mean log power of the 14 turbines.
- T01 at 2016-09-05 15:00 has availability 0.528 and 167 kW at 7.9 m/s. a "P output externally reduced" event starts at 15:06:55, and the availability rule drops the row first.

## histories

every development history (from 2017-04-01, 07-01, 10-01 and 2018-01-01) and every confirmation-period history (from 2020-01-01, 04-01, 07-01 and 10-15) keeps at least 300 training timestamps, and every threshold fit is included too (`runs/<study>/manifests/penmanshiel_summary.csv`).
