# Hill of Towie

RES and TRIG's Hill of Towie wind farm in Moray, Scotland: 21 Siemens SWT-2.3-VS-82 turbines, T01 to T21, with an 82 m rotor, a 59 m hub and 2300 kW rated. commercial operation started on 2012-05-12. the farm sits on hilly ground.

- source: Zenodo record "Hill of Towie open dataset", version 2.1.0, DOI 10.5281/zenodo.22662930 (concept DOI 10.5281/zenodo.14870021), released by RES on behalf of TRIG. 2.1.0 changed documentation only; its data files equal 2.0.0's.
- license: CC BY 4.0, provided as is.
- companion code: RES's hill-of-towie-open-source-analysis at commit 03bb9ccee18b56b9c32595cbb7298a4244951769, for the northing file and the exclusion periods.
- raw files: the yearly SCADA zips 2023 to 2025, the shutdown-duration zip and the metadata files, each listed with bytes and MD5 in `src/windfold/configs/farms/hill_of_towie.toml`.

## what we read

the 10-minute SCADA tables, one row per turbine and interval start in UTC: 3,260,973 rows for 21 turbines, from 2022-12-31 23:50 to 2025-12-31 23:50. the raw files label each interval by its end; the reader moves the label to the start (by hand: T01's raw 02:40 line is our 02:30 row).

| column | SCADA field |
|---|---|
| power_kw | tblSCTurGrid.wtc_ActPower_mean |
| speed, speed_std | tblSCTurbine.wtc_PrWindSp_mean, _stddev (the primary anemometer) |
| wind_direction, its min and max | tblSCTurbine.wtc_ActualWindDirection_mean, _min, _max |
| nacelle_position, its min and max | tblSCTurbine.wtc_NacelPos_mean, _min, _max |
| ambient_temperature | tblSCTurTemp.wtc_AmbieTmp_mean |
| state_code_min, state_code_max | tblSCTurbine.wtc_WTOperSt_min, _max |
| running_s | tblSCTurFlag.wtc_ScInOper_timeon |
| turbine_error_s, environment_stop_s, communication_stop_s, grid_stop_s | tblSCTurFlag stop timers |
| setpoint_end_kw | tblSCTurbine.wtc_PowerRef_endvalue |

the events are 1,072,842 alarm-log entries and 24,963 shutdown-log entries. the reader drops station 91's alarms, which come from the park controller, not a turbine.

the layout is the readers' `data/tables/hill_of_towie/layout.parquet`, copied to `data/prepared/hill_of_towie/layout.parquet`. its east and north meters reproduce the WGS84 geodesic distance of every turbine pair to the millimeter. the rotor diameter (82 m) comes from the turbine config.

## quirks

- the monthly files overlap by one timestamp at each month boundary; the reader keeps each copy once.
- 29,729 rows come from intervals when the controller ran on its secondary anemometer (T21 6,226, T02 5,535). the reader makes their speed and spread missing, so step 4 drops those that reach it.
- a farm-wide outage runs from about 2025-09-10 14:50 to 09-23 11:40, and T02's to 09-30. it falls inside the history test window (2025-06-08 to 2025-10-28 13:00).
- the operating state code (wtc_WTOperSt) has no published meaning: the data dictionary calls it "turbine operating state code", the metadata says nothing and the companion code never reads it. the data shows what the codes do:
  - 0 is production: 97% of its intervals make power, with the turbine running all 600 s.
  - 3 is waiting for wind: no power (median −13.6 kW), no running time, every stop timer at zero and a median own speed of 2.30 m/s (90% under 3.09).
  - an interval holding 0 and 3 is a cut-in or cut-out: 88% of them contain a "Large generator Cut-in" alarm (code 20) and 82% a "Fast cut-out of generator" alarm (code 25), at a median own speed of 3.72 m/s. 97% of the intervals with a cut-in or cut-out alarm hold only 0 and 3.
  - 1 and 2 are start-up states: 95% of the intervals holding 0 and 1 contain a cut-in alarm (all of those holding 0 and 2), at 4.5 to 4.6 m/s; intervals holding 1 or 2 with 3 sit at 2.5 to 2.8 m/s, with no power and no stop timer.
  - 4, 5, 6, 8, 9 and 11 are stops: most of their intervals carry turbine-error or environment stop timers, and 6 sits at a median of 23.9 m/s (the high-wind stop). 7 appears when no other turbine produces.

  step 3 keeps only intervals in state 0 throughout. 722,589 of the intervals it drops hold only states 0 and 3.
- direction means were sometimes averaged linearly across north: the interval's minimum is near 0°, its maximum near 360° and its mean far from both. 44,320 wind-direction rows and 71 nacelle-position rows like this are dropped at step 4.
- the direction minimum and maximum sometimes read −1.0. 8,102 rows have a minimum of −1, and 4,703 of them a maximum of −1 too; in those the mean is exactly 359.0 and no turbine makes power. the linear-mean check treats −1 as missing. where the missing value decides the check (a minimum of −1, a maximum near 360° and a mean far from north, 92 rows) the row is dropped, as are rows whose minimum or maximum is missing in the file when it decides the check; the rest are decided by their mean or maximum.
- logged icing appears as alarm codes 102 (254 events in 2023 to 2025, each a point in time) and 8230 (208 events, median 2 hours). an iced turbine leaves state 0, so no logged icing interval reaches step 8.
- the free-stream turbines do not see the same wind. on the calibration window, at reference speeds of 5 to 9 m/s, each turbine's speed over the median of the other free-stream turbines ranges from 0.872 (T18) to 1.215 (T19). power mostly follows: T21 reads 1.111 and makes 1.381 of their power where the curve predicts 1.405, T15 0.896 and 0.708 (0.690), T06 0.942 and 0.828 (0.823). some turbines follow only in part (T19 makes 1.363 where the curve predicts 1.817), so part of their difference may be the sensor. the anemometer factors below therefore reach only the reference speed.
- northing: we apply the provider's dynamic-yaw design file alone, from 2019-10-14 23:50. it holds a complete per-turbine schedule through the 2025 recalibrations (T03 −177° from 2025-08-07). over 2023-06 to 2024-09 it agrees within 0.5° with the optimized file on 19 turbines and is 1.58° higher on T05 and 0.80° on T15. our data sides with it: against the median of the other turbines, T05 sits −0.29° with the design file and −2.05° with the optimized one, T15 −0.70° and −1.60°, and the mean absolute deviation over all turbines is 0.40° against 0.58°. wake dips seen in T05's own direction agree (−2.0° against −4.0°, with the farm median at −2.2°).

## what we drop and why

every dropped row is counted once, under the first rule that drops it (`data/prepared/hill_of_towie/ledger.csv`).

| step | rule | rows dropped | rows kept | timestamps kept |
|---|---|---|---|---|
| 1 | read | 0 | 3,260,973 | 155,972 |
| 2 | windows | 649,419 | 2,611,554 | 124,945 |
| 3 | power_missing | 243 | 2,611,311 | 124,945 |
| 3 | state_code | 812,055 | 1,799,256 | 104,214 |
| 3 | running_time | 124 | 1,799,132 | 104,214 |
| 3 | stop_time | 0 | 1,799,132 | 104,214 |
| 3 | stop_event | 2 | 1,799,130 | 104,214 |
| 3 | curtailment_event, derate_event, unknown_event | 0 | 1,799,130 | 104,214 |
| 3 | binding_setpoint | 79,949 | 1,719,181 | 101,064 |
| 3 | power_floor | 5 | 1,719,176 | 101,060 |
| 4 | speed_missing | 15,177 | 1,703,999 | 101,046 |
| 4 | direction_missing, out_of_range, frozen_row, stuck_speed, stuck_direction | 0 | 1,703,999 | 101,046 |
| 4 | linear_mean_wind_direction | 44,320 | 1,659,679 | 100,943 |
| 4 | linear_mean_nacelle_position | 71 | 1,659,608 | 100,941 |
| 5 | power_curve | 11,683 | 1,647,925 | 100,540 |
| 6 | exclusion (RES farm-wide curtailment) | 115,296 | 1,532,629 | 93,686 |
| 6 | exclusion (pitch tune-up) | 357 | 1,532,272 | 93,647 |
| 7 | manual_yaw | 0 | 1,532,272 | 93,647 |
| 8 | icing_event | 0 | 1,532,272 | 93,647 |
| 9 | wake_source | 214,584 | 1,317,688 | 90,681 |
| 10 | reference | 154 | 1,317,534 | 90,583 |
| 10 | speed_window | 457,055 | 860,479 | 55,329 |
| 11 | valid_turbines | 433 | 860,046 | 54,896 |

- windows: rows before 2023-06-01 and from 2025-10-28 13:00, when dynamic-yaw testing with wake steering starts.
- status: state codes, running time under 599 s, stop timers, alarm codes for stops and the shutdown log. the binding setpoint uses the setpoint at the end of the interval and of the one before, so a cap that changed inside the interval also counts. the state code leaves the power floor almost nothing.
- power curve: bounds around the curve fitted on the calibration window, both on measured speeds. 11,683 rows (0.70%) fall outside. applying the anemometer factors to each turbine's own speed would have added about 51,000 drops, mostly on T19, T21 and T18, whose power follows their measured speed.
- exclusions: the ten RES windows of farm-wide curtailment, start inclusive and end exclusive, and the pitch tune-up of 2024-03-14 07:00 to 18:00, which drops the 357 of its 1,386 rows that reach it (the others were already dropped, 1,029 by the state code).
- direction: the provider's northing, then the farm rotation. there are no per-turbine offsets here.
- icing: only logged icing events count, and none reaches this step. a cold-and-low-power icing rule would fire on 9.5% of the 243,290 cold rows that reach it and on 9.5% of the 1,288,982 warm rows (measured speeds) and on no logged icing; it finds scatter, not ice.
- waked low-wind stops: step 12 counts, per speed bin, the waked rows that step 3 dropped as stopped at low wind. here the state code drops them before the power floor can, so they count through the low-wind states, set to the states 1, 2 and 3 found above: 33,307 at reference speeds 5 to 6.5 m/s, 8,592 at 6.5 to 8 and 2,774 at 8 to 10 (with 3 alone, 32,665, 8,375 and 2,684). these include cut-in intervals that made some power, which the power floor alone would keep at the other farms.

prepared rows (`data/prepared/hill_of_towie/rows.parquet`):

| period | rows | timestamps | waked share | valid turbines per timestamp |
|---|---|---|---|---|
| calibration (2023-06-01 to 2023-08-31) | 91,975 | 5,962 | 0.567 | 15.43 |
| development (2023-09-01 to 2025-10-28 13:00) | 768,071 | 48,934 | 0.582 | 15.70 |

the speed bins hold 285,193, 289,201 and 285,652 rows (5 to 6.5, 6.5 to 8 and 8 to 10 m/s), waked in 53.9%, 59.3% and 60.8% of them. 0.28% of timestamps (152) have exactly one free-stream turbine. 96% of timestamps are steady. the target sums to zero at every timestamp (largest |Σy| 2.3e-14). T04, T08, T10 and T12 sit inside the farm and are waked in 90 to 99% of their rows.

## calibration from the data

- anemometer factors, fitted on the calibration window as at Kelmarsh, with at least 1,000 qualifying timestamps per turbine. T04 (102), T08 (51), T10 (484) and T12 (384) are rarely free-stream there and keep 1. they reach the reference speed only.

  | turbine | T01 | T02 | T03 | T05 | T06 | T07 | T09 | T11 | T13 | T14 | T15 | T16 | T17 | T18 | T19 | T20 | T21 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | factor | 1.005 | 1.069 | 0.995 | 0.962 | 1.057 | 0.920 | 1.005 | 1.016 | 0.939 | 0.969 | 1.120 | 1.120 | 1.028 | 1.158 | 0.832 | 1.050 | 0.906 |

  on the prepared rows the reference speed differs from the median of the measured free-stream speeds by a median of +0.021 m/s (5 to 95%: −0.49 to +0.62, at most 1.9), and 130,542 rows (15.2%) sit in another speed bin.
- farm rotation: +2.08°, the median wake-dip offset of 53 turbine pairs 2 to 10 diameters apart on the calibration window. a check over 2016 to 2026 with the provider's corrections found the dips 2.3° from the bearings in the farm direction, the same rotation (`runs/inspect/hill_of_towie/wake_dips.pdf`). one pair, whose downstream turbine makes twice the upstream's power at every direction, has no dip and is left out.
- power curve: fitted on the calibration window, on measured speeds.
- temperature: readings outside −40 to 50 °C would be made missing; none is.

## checks by hand

- T01 at 2024-01-01 02:30: the raw 2024-01-01 02:40 lines of station 2304510 hold 290.2248 kW (tblSCTurGrid) and 6.553823 m/s (tblSCTurbine), as the readers' row and the prepared row do. the row is free-stream. the reference speed of 6.3353 m/s is the median of the six free-stream turbines after their factors (T01 6.5852, T02 6.3452, T03 5.6416, T07 6.2177, T09 6.3255, T13 7.6858), and its y of 0.267173 equals log power minus the mean log power of the 20 turbines kept at that timestamp.
- T11 is missing from that timestamp: it makes 406 kW in state 0 throughout, but its controller was on the secondary anemometer, so its speed is missing and step 4 drops it.

## histories

two splits fall short and are left out (`runs/development/manifests/hill_of_towie_summary.csv`):
- the 7-day history from 2023-12-01 keeps 213 training timestamps, under the 300 needed.
- the 91-day threshold fit from 2024-06-01 has no threshold timestamps: its threshold days (from 2024-08-12) fall inside the RES exclusion of 2024-08-09 to 09-28.

every other history and threshold fit is included.
