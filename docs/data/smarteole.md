# SMARTEOLE

ENGIE Green's open dataset from the third SMARTEOLE field campaign at Sole du Moulin Vieux, France: seven Senvion MM82 turbines, SMV1 to SMV7, with an 82 m rotor, an 80 m hub and 2050 kW rated. wake steering was applied on SMV6, switched on and off in turns.

- source: Zenodo record "SMARTEOLE Wind Farm Control open dataset", version 1.0, DOI 10.5281/zenodo.7342466 (concept DOI 10.5281/zenodo.7342465), Duc and Simley 2022. this is the only version.
- license: Etalab Open Licence 2.0.
- acknowledgement, as the record asks: "The dataset used in this research has been obtained by ENGIE Green in the scope of French national project SMARTEOLE (grant no. ANR-14-CE05-0034)". the text should also cite Simley et al. 2021, Wind Energy Science 6, 1427-1453, DOI 10.5194/wes-6-1427-2021.
- raw files: `SMARTEOLE-WFC-open-dataset.zip`, listed with bytes and MD5 in `src/windfold/configs/farms/smarteole.toml`.

## what we read

the 1-minute SCADA, one row per turbine and minute start in UTC: 942,627 rows (7 × 134,661 minutes) from 2020-02-17 16:30 to 2020-05-24 23:59.

| column | SCADA field |
|---|---|
| power_kw | active_power_{i}_avg |
| speed, speed_std | wind_speed_{i}_avg, _std |
| wind_direction | wind_direction_{i}_avg |
| nacelle_position | nacelle_position_{i}_avg |
| ambient_temperature | temperature_{i}_avg |
| derated | derate_{i} |

the control log becomes 1,333 events on SMV6: 1,321 runs with steering active and 12 with its state unknown. control labels are minute starts, and steering switches about 8 s after the hour.

the layout is the readers' `data/tables/smarteole/layout.parquet` (from the record's coordinates file; the 80 m hub height is in the farm config), copied to `data/prepared/smarteole/layout.parquet`. its east and north meters reproduce the WGS84 geodesic distance of every turbine pair to the millimeter.

## quirks

- five temperature readings are sensor glitches: 932.08, 152.73 and 63.133 °C, 2.46e18 and 4.24e16 °C (SMV1 on 2020-05-15, SMV4 on 05-20 and 05-23). readings outside −40 to 50 °C are made missing before the minutes are averaged; the rows stay.
- the record ships two correction files, and we apply neither:
  - `correction_factors_SMV1237` scales the average of SMV1, 2, 3 and 7 to match SMV6. it builds a reference signal for Simley et al.'s energy-gain analysis; it is not a per-turbine sensor calibration.
  - `NTF_SMV6` maps SMV6's nacelle speed alone to the lidar's free stream. applying it would put one turbine on a different speed basis from the other six.
- there is no calibration window, so the anemometer factors are 1, the power curve is the published one and there is no farm rotation from wake dips. the provider's directions are used as read.

## what we drop and why

every dropped row is counted once, under the first rule that drops it (`data/prepared/smarteole/ledger.csv`). the first four lines count minutes; from minutes_per_window on, rows are 10-minute windows.

| step | rule | rows dropped | rows kept | timestamps kept |
|---|---|---|---|---|
| 1 | read | 0 | 942,627 | 134,661 |
| 2 | windows | 0 | 942,627 | 134,661 |
| 3 | derated | 39,293 | 903,334 | 134,661 |
| 6 | control | 512,069 | 391,265 | 58,423 |
| 6 | minutes_per_window | 2,248 | 38,280 | 5,762 |
| 3 | power_missing, stop_event, curtailment_event, derate_event, unknown_event | 0 | 38,280 | 5,762 |
| 3 | power_floor | 12,826 | 25,454 | 4,133 |
| 4 | speed_missing, direction_missing, out_of_range, frozen_row, stuck_speed, stuck_direction | 0 | 25,454 | 4,133 |
| 5 | power_curve | 151 | 25,303 | 4,131 |
| 7 | manual_yaw | 0 | 25,303 | 4,131 |
| 8 | icing_event | 0 | 25,303 | 4,131 |
| 9 | wake_source | 830 | 24,473 | 4,079 |
| 10 | reference | 11 | 24,462 | 4,073 |
| 10 | speed_window | 6,997 | 17,465 | 2,715 |
| 11 | valid_turbines | 7 | 17,458 | 2,708 |

- derated: minutes the turbine flags as derated, or with the flag unknown.
- control: every turbine's minutes while steering on SMV6 is active or unknown, and for the 20 minutes after it ends, so the wakes downstream have settled. steering ran about half the time, so this drops more than half the minutes.
- minutes_per_window: a window needs at least 7 complete minutes (power, speed and direction all present). windows are 10-minute means, with angles averaged on the circle and the speed spread pooled within and between minutes.
- power curve: 151 windows (0.6%) fall outside the bounds around the published guaranteed curve.
- wake sources and inputs: as at Kelmarsh. the reference speed must lie between 5 and 10 m/s.
- waked windows stopped below the power floor: 180 at reference speeds 5 to 6.5 m/s, 51 at 6.5 to 8 and 6 at 8 to 10.

prepared rows (`data/prepared/smarteole/rows.parquet`): 17,458 rows at 2,708 timestamps, all development; 26% waked, 6.45 valid turbines per timestamp, 72% of timestamps steady. the speed bins hold 6,987, 5,927 and 4,544 rows (5 to 6.5, 6.5 to 8 and 8 to 10 m/s), waked in 25.7%, 29.4% and 22.8% of them. 19.2% of timestamps (519) have exactly one free-stream turbine: wind along the north-south line leaves only the first turbine free, and its own anemometer then gives the reference speed. the target sums to zero at every timestamp (largest |Σy| 7e-15).

## checks by hand

- SMV1 at 2020-02-18 12:40: the ten raw minutes 12:40 to 12:49 in `SMARTEOLE_WakeSteering_SCADA_1minData.csv` are complete and not derated, with no control event in the window or the 20 minutes before it. their mean power is 1437.1354 kW and mean speed 10.1687 m/s, as in the prepared row. all seven turbines are free-stream at that timestamp, so the reference speed is the median of the seven, 9.8116 m/s (SMV5), and SMV1's y of 0.037227 equals log power minus the mean log power of the seven.

## histories

two histories fall short and are left out (`runs/development/manifests/smarteole_summary.csv`): the 7-day history from 2020-02-17 keeps 177 training timestamps and the 14-day one 293, under the 300 needed. the 30- and 60-day histories and both threshold fits are included.
