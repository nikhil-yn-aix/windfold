# reproduce

every table and figure of the study comes from `windfold` commands run in the repository's root, with one exception: the monitoring report of item 58 of the [study plan](study_plan.md) came from its reference script, and `windfold secondary monitoring_threshold` reproduces it. the commands keep their state in two folders, `data` and `runs`, in the working directory; `--data` and `--runs` put them elsewhere.

## what you need

- Linux on x86-64, the only platform the lockfile resolves for, with Python 3.12 and [uv](https://docs.astral.sh/uv/) (the checks use uv 0.13.7).
- torch 2.13.0 from one of two extras that exclude each other: `cpu`, or `gpu`, built for CUDA 12.6. AutoGluon 1.6.3 comes with the `automl` dependency group, and pytest, ruff and pyright with `dev`.
- no GPU is required. the graph networks, their pretraining and peer AutoML use a CUDA GPU when one is usable and run on CPU otherwise. peer AutoML trains within a time budget, so its fits depend on the machine's speed, and seeded models can differ slightly between torch builds.

every job's fingerprint covers the code it runs and the versions of the packages it reaches, torch's build tag among them. keep the packages and the code fixed while a study runs: a change gives the jobs it touches, and everything after them, new fingerprints, so they run again.

## install and check the configs

```
uv sync --locked --extra cpu --group automl
uv run windfold configs
```

use `--extra gpu` instead on a machine with an NVIDIA GPU.

add `--group dev` to run the tests with `uv run pytest`. `windfold configs` loads and checks every packaged config. they live in `src/windfold/configs`:

- `protocol.toml`: every fixed setting of the study: processing, splits, statistics, monitoring, the decision rule, seeds, the physics search, the engine and shared correction, the coverage score, the named sensitivities, the planned tests' endpoints and the tuning menus' search spaces.
- `models.toml`: the 23 models and windfold's 12 variants, with their terms, learners and inputs.
- `farms/<farm>.toml`: each dataset's DOI and license, windows, histories, channels, rules, exclusions and every raw file with its size and MD5.
- `turbines/<type>.toml`: power and thrust curves, each number with its source.
- `studies/{smoke,development,confirmatory}.toml`: each study's farms, models and sensitivities.
- `menus/{gbm,graph}.toml`: the 30-configuration tuning menus, drawn once by `windfold menus`, which refuses to draw them again.
- `resources.toml`: the folders and how the runner sizes, schedules and pauses its jobs.
- `report.toml`: the style of the report's figures and tables.

`fetch`, `run`, `status`, `release`, `freeze`, `secondary`, `analyse` and `figures` take `--configs <folder>` to read a copy of this folder instead. a changed config changes the fingerprints of the jobs that read it.

## fetch the data

```
uv run windfold fetch
```

downloads every raw file the farm configs list into `data/raw/<farm>/`, from its pinned URL or by its name in the dataset's Zenodo record, and checks each file's size and MD5. a file already present is never downloaded again, even when it differs; delete it and fetch again. `windfold fetch kelmarsh` fetches one farm, and the data jobs check every raw file again before reading it.

## run the studies

`windfold run <studies>` plans every job of the named studies and their inputs, runs those not done in dependency order and resumes where it stopped. `--plan` lists the jobs and how many are done, `--detach` runs the runner in the background with its output in `runs/logs/detached.log`, `--cpu-jobs` caps the CPU slots and `--keep-waiting` keeps it replanning when nothing can run. the whole sequence:

```
uv run windfold run development confirmatory
uv run windfold release --farm kelmarsh --farm penmanshiel --farm hill_of_towie --farm smarteole
uv run windfold run smoke
uv run windfold release --family tuning --family pretraining --family automl_budget \
  --family engine --family gbm --family graph --family pretrained_graph --family automl \
  --family evaluation
uv run windfold run development confirmatory
uv run windfold freeze first
uv run windfold run development confirmatory
uv run windfold freeze second
uv run windfold run development confirmatory
```

1. the first run prepares the data: the readers' tables, the prepared rows, the inspection plots, the physics tables, the shared correction's design and the split manifests. model jobs wait.
2. check each farm's data in `runs/inspect/<farm>/` against its [card](data/), then release the farms. `windfold release` records released farms and families in `runs/release.toml`.
3. the smoke study runs every family on the released farms under `runs/smoke/`, with stand-ins the real studies never use: each learner's first menu configuration, an untrained pretrained graph and a short AutoGluon budget.
4. release the families once their smoke jobs pass.
5. the second run fits what the calibration windows decide, in `runs/calibration/` (tuning, pretraining and AutoGluon budgets), then the development study with its 14 sensitivities and its report. no confirmation-period fit is planned yet.
6. `freeze first` writes `runs/confirmatory/freeze_first.json`.
7. the third run makes the confirmation-period fits that match the first record, and no others.
8. `freeze second` writes `runs/confirmatory/freeze_second.json`. the confirmation period's evaluation is planned only once it exists.
9. the last run scores the confirmation period and writes the decision and the report of both studies in `runs/report/`, which needs both studies in one plan.

each of the 14 named sensitivities changes one setting and refits only the models that setting can change. the processing changes (no direction corrections, speeds of 4 to 12 m/s, Hill of Towie after its 2024 pitch upgrade) and the split changes (buffers of 0 or 10 degrees instead of 5, 60-degree sectors) refit 17 models: the no-wake model, the power curve, the direction lookup, the cyclic GAM, both linear models, PyWake and PyWake with discrepancy at both inflows, windfold and six of its variants. the correction changes (a window of 20 or 40 diameters along or 2 across, a piecewise-constant correction) refit wind-frame linear and windfold. `fixed_shrinkage` holds the prior scales of the six Gaussian comparison models that fit them at windfold's value. `fixed_physics_scale` holds the physics weight near one in PyWake with discrepancy at both inflows, windfold and five of its variants. `block_days_1` and `block_days_14` only rerun the bootstrap with other block lengths. no sensitivity refits the GBMs, graph networks or peer AutoML. the `development` study runs all fourteen and the `confirmatory` study only `fixed_shrinkage`.

## the freeze records

the confirmation period runs between two records in `runs/confirmatory/`:

- `freeze_first.json` lists every confirmation-period fit by job ID with its fingerprint, which covers the code, configuration, data, splits and calibration-window outputs the fit rests on, together with the sha256 of each calibration-window output those fits read and the commit. the published record holds 1,618 fits and 24 calibration-window outputs. the two records it replaced are kept beside it as `freeze_first_replaced_<time>.json`.
- `freeze_second.json` holds the coverage map and simple-predictor choice fitted on the development window, and the sha256 of every output of the frozen fits: 7,842 files, the 3,716 prediction tables with their records, 384 training curves and 26 AutoGluon records.

`windfold freeze first` or `second` refuses to overwrite a record; `--again` replaces it and keeps the old one beside it. git tracks all four records, so a clone starts with them in place. confirmation-period fits whose fingerprints match the first record are then planned without `freeze first`, and the evaluation refuses any output whose sha256 differs from the second record's. a rerun with other packages or code gets new fingerprints, so its confirmation-period fits drop out of the plan until both records are written again with `--again` at steps 6 and 8.

to check every frozen output against the second record:

```
uv run python - <<'EOF'
import hashlib, json, pathlib
runs = pathlib.Path("runs")
record = json.loads((runs / "confirmatory/freeze_second.json").read_text())
bad = [p for p, h in record["predictions"].items()
       if hashlib.sha256((runs / p).read_bytes()).hexdigest() != h]
print(len(record["predictions"]), "files,", len(bad), "differ")
EOF
```

## the runs folder, resuming and pausing

one runner works on a runs folder at a time and keeps everything it needs to resume there. `run.lock` refuses a second runner on the same runs or data folder. `jobs.jsonl` is the append-only job log, one line per event (queued, skipped, started, retried, done, failed or dropped) with the job, its family, farm and fingerprint and any error. running jobs write under `staging/`, and their outputs move into place only after a clean exit with every output hashed and finite. every output then has a sidecar, `<file>.sidecar.json`, with the job, its fingerprint and the file's sha256 and size. `logs/` holds the runner's log, the status view and each job's output and traceback.

to resume after a stop or a kill, run the same command again. the new runner stops anything the old one left running, empties `staging/` and checks every job of the plan: a job counts as done only when every output exists, its sidecar names the job's current fingerprint and the file's sha256 matches. a failed job is retried once. after a second failure the jobs that need it wait, and the next run gives it two more attempts.

a family pauses when enough of its jobs fail for good (the thresholds are in `resources.toml`), or at once when a job writes a value that is not finite. the reason is in `paused/<family>`; fix the cause and delete that file to resume the family. a single job run with `--job <id>` ignores pauses. `windfold status` shows whether a runner is running and the jobs of the latest run by family and farm, with alerts such as failed jobs with their last error, paused families, AutoML jobs waiting for a GPU and jobs left over with no runner.

## the tests set before the confirmation period was scored

```
uv run windfold secondary hybrid_gbm --study confirmatory --farm penmanshiel --fit
uv run windfold secondary restricted_windfold --study confirmatory --farm penmanshiel --fit
uv run windfold secondary monitoring_threshold --study confirmatory --farm penmanshiel
```

these score the combined GBM, the tighter variant and the recalibrated alarm threshold of [results.md](results.md). each refuses confirmation-period data until the second freeze record exists and writes `runs/<study>/secondary/<name>/<farm>.parquet`, where the names are `hybrid_gbm`, `windfold_restricted` and `monitoring_threshold` (`--out` writes elsewhere). `--study development` with a development farm gives the development tables.

the first two fit models. `--fit` makes the missing fits first (`--workers` sets how many run at once) and puts them under `predictions/` beside the table; without it the command only scores. a fit is reused only while its record's provenance matches: a digest of its code, configuration, seed, split and the sha256 of its inputs. each contrast comes in MSE and in skill points, with the no-wake model's MSE on the same rows dividing it inside every bootstrap replicate. the published confirmation-period tables were written before the skill points existed; scoring the same fits again without `--fit` and with `--out` adds them and gives the published MSE columns exactly.

`monitoring_threshold` fits nothing. it reads the monitors' threshold predictions and the monitoring reductions and writes one long table per farm. the table holds recall and alarm rates at every history under three rules, the standard threshold, the recalibrated one and an intermediate rule that item 58 of the study plan also reports, with intervals at 30 days (60 on SMARTEOLE). it also holds windfold's recall gains and alarm-rate margins and the outcome of both checks. it reproduces the reference script's report, `runs/confirmatory/secondary/monitoring_threshold/report.md`, exactly in every rate, gain, margin and p-value. in that report's 30-day tables the columns after the standard threshold's are labeled a, b and c but hold the intermediate rule and then the recalibrated one.

## exploratory analyses

`windfold analyse <name>` writes one analysis behind a statement of the study to `runs/<study>/analyses/<name>.json`, with the code commit, the study, the farms analyzed and the folders read (`--out` writes elsewhere and `--farm` limits the farms). these analyses test nothing. `--study` defaults to development; `comparison_ranks`, `default_physics`, `energy_ratio_error`, `energy_ratio_intervals`, `interval_score`, `pair_skill`, `practical_scale`, `secondary_accuracy` and `sector_gains` can also read the confirmation period once its second freeze record exists, and the rest read development data only.

- `arc_variants` (refits): the no-wake model, PyWake at measured inflow and windfold refitted on the held-out sectors with an hour of time buffer around their test rows and on the sector grid shifted by half a sector, and windfold's margin over PyWake under each.
- `comparison_ranks`: Kendall's τb between the random, time-blocked and held-out-sector rankings, over windfold and the comparison models, the comparison models alone and all models.
- `default_physics` (refits): PyWake at measured inflow with PyWake's published default wake parameters in place of the calibrated ones, on every held-out sector, and calibrated PyWake and windfold each against it.
- `direction_spread`: how closely the free-stream turbulence follows the 10-minute scatter of wind direction over the unwaked turbines at Kelmarsh and Penmanshiel, from the raw archives.
- `energy_ratio_error`: each model's mean absolute error of predicted against observed energy ratio on held-out sectors, over turbine and 5-degree bucket cells with enough rows and weighted by their rows.
- `energy_ratio_intervals`: the errors of `energy_ratio_error` with 95% intervals from 2,000 replicates of the farm's paired bootstrap of 7-day blocks, which also redraws the seeds of seeded models, and windfold's error minus each model's.
- `inflow_smoothing` (refits): the measured turbulence replaced by its averages over longer windows, and the change in windfold's held-out skill.
- `inflow_structure`: how much of the measured turbulence's variance lies between direction sectors, speed bins and the diurnal control's cells, and how many timestamps that control averages.
- `interval_score`: the coverage and interval score of the Gaussian models' 90 and 99% intervals on waked rows of held-out sectors and time-blocked folds, and windfold minus each other Gaussian model in those and in CRPS.
- `kernel_geometry`: what windfold's shared correction adds to the physics on the turbine pairs that occur, per windfold variant, and how much it varies between held-out sectors.
- `lidar_turbulence`: SMARTEOLE's free-stream and nacelle turbulence against the lidar at hub height, from the raw archive.
- `linear_clipping`: the linear pair with identity linear's held-out predictions clipped to the 0.1 to 99.9% range of the development targets.
- `local_block`: windfold with and without local cells on held-out sectors: physics weights, correction size, each split's skill gaps and the mean waked residuals.
- `pair_skill`: skill differences on held-out sectors for seven model pairs.
- `practical_scale`: windfold against PyWake at measured inflow and the strongest learned model in MSE, relative MSE, skill points and kW, pooled and per sector.
- `same_row_skill`: each compared model's skill under time-blocked validation and on held-out sectors, both on the waked rows held-out validation tests and against the no-wake model's time-blocked error on those rows.
- `secondary_accuracy`: skill, MSE, mean absolute kW error and CRPS on waked rows of the held-out sectors for the combined GBM and the tighter correction, the two models fitted by `windfold secondary`, scored with the farm's planned bootstrap as the study scores its own models.
- `sector_gains`: windfold minus PyWake at measured inflow on each held-out sector beside its share of waked rows, the Spearman correlation between the two and the sectors windfold leads, trails or does not resolve.
- `sensor_count`: windfold at measured minus diurnal inflow, by how many unwaked turbines give each waked row's turbulence reading.
- `strict_alignment` (refits): the farm rotation fitted again without each held-out sector and its shift; with `--refit`, the physics rebuilt at the shifted directions and the change in the held-out skill of windfold and PyWake at measured inflow.
- `strict_calibration` (refits): every wake family recalibrated without each held-out sector, and the change in the held-out skill of windfold and PyWake at measured inflow.
- `transfer_ceiling`: how much of the held-out error of PyWake at measured inflow the shared terms recover on each sector's own rows and from the other sectors, and how much windfold realizes, with the penalty chosen per sector with its score left out and with its rows left out too.
- `turbine_levels`: the GBM pair on held-out sectors split into each turbine's level and the rest, with the same parts under time-blocked validation.
- `validation_choice`: the share of bootstrap replicates in which time-blocked validation chooses each model, and how its ranking agrees with the held-out ranking.

their intervals come from a paired bootstrap of 7-day blocks of the development window, 2,000 replicates seeded per analysis and farm, so the same inputs give the same numbers. `sensor_count` resamples each stratum's own blocks, 1,000 replicates, and `practical_scale` and `validation_choice` use the study's own bootstrap of 10,000 replicates with its seed redraws.

the analyses marked as refitting keep their refits under `runs/development/analyses/<name>/` (`--work` puts them elsewhere), refuse to run while refits are missing and make them with `--refit`, with `--workers` processes for the physics. a finished split is kept, so a stopped run resumes, and a refit is reused only while its provenance matches. farms can run side by side, each with `--farm`, before one run without `--farm` writes the full result:

```
uv run windfold analyse strict_calibration --refit --farm kelmarsh
uv run windfold analyse strict_calibration
```

## figures

the runner draws the report's tables and most of its figures in `runs/report/`, and the development study's alone in `runs/development/report/`. `uv run windfold figures` draws the figures that read the outputs of `windfold secondary` and `windfold analyse` once those exist, checks each by the report's style rules and writes it to `runs/report/figures/` as a vector PDF with PNG previews (`windfold figures <name>` draws one, `--out` writes elsewhere). the two diagrams of the model and the validations are drawn apart from the runner:

```
uv run python -m windfold.report.diagrams <folder> windfold splits
```

the model diagram reads Penmanshiel's prepared layout in `data/prepared/`, and the validations diagram reads only the configs.

## where the outputs land

- `data/raw/<farm>/`: the raw files.
- `data/tables/<farm>/`: the readers' tables, `layout.parquet` among them.
- `data/prepared/<farm>/`: `rows.parquet`, `layout.parquet`, `ledger.csv` with every dropped row by rule, `provenance.json`, `physics.parquet` and `inflow_physics.parquet` with their fit records, the inflow controls' tables, `kernel_design/` with the shared correction's design and its `permuted/` copy and one folder per sensitivity that changes the processing or the correction.
- `runs/inspect/<farm>/`: the inspection plots, `rows_kept.csv` and `rows_dropped.csv`.
- `runs/calibration/`: tuning, pretraining and AutoGluon budgets for each farm.
- `runs/<study>/manifests/`: the split manifests and `<farm>_summary.csv`.
- `runs/<study>/predictions/<farm>/<model>/`: one prediction table and record per split and seed. the code and its output files call a held-out sector an arc, named by the direction where it starts: `arc_000_single.parquet` holds the sector from 0 to 30 degrees held out alone, and the `_opposite` and `_control` splits also withhold the opposite sector or a matched set of other directions. `runs/<study>/curves/` holds the graph networks' training curves and `runs/<study>/automl/` the AutoGluon records.
- `runs/<study>/reductions/<farm>/`: the daily sums the results are built from.
- `runs/<study>/results/`: the study's tables and decision, with one folder per farm.
- `runs/<study>/sensitivities/<name>/`: each sensitivity's predictions, reductions and results.
- `runs/<study>/secondary/` and `runs/<study>/analyses/`: the tests set before the confirmation period was scored and the exploratory analyses.
- `runs/report/`: the tables and figures of both studies.

the numbers of [results.md](results.md) come from these files:

- skill by model and validation: `runs/report/tables/main_results.csv`, and with kW and CRPS, `runs/<study>/results/accuracy.parquet`. CRPS is defined only for the Gaussian models, and the other models' CRPS rows read 0 but mean undefined (item 62 of the study plan).
- contrasts with windfold: `runs/<study>/results/comparisons.parquet`; windfold's variants: `runs/report/tables/ablations.csv`.
- the four contrasts set before the confirmation period was scored: `runs/confirmatory/results/repair.parquet`.
- rank agreement over all 34 models scored under every validation: `runs/<study>/results/rank_agreement.parquet`; over the 22 models of the comparison: `runs/<study>/analyses/comparison_ranks.json`.
- the planned tests: outcomes in `runs/report/tables/hypotheses.csv` and `runs/confirmatory/results/decision.csv`, and every component in `runs/<study>/results/hypotheses.csv`.
- 365-day non-inferiority and interval coverage: `runs/<study>/results/<farm>/noninferiority.parquet` and `calibration.parquet`.
- monitoring: `runs/<study>/results/monitoring.parquet`. windfold's monitor against three monitors matched to its information (the identity GBM and PyWake at measured inflow and windfold at constant inflow), outside the planned tests, is in `matched_monitoring.parquet` beside it.
- the split of held-out error, recalibrated physics, practical units and per-sector gains: `transfer_ceiling.json`, `strict_calibration.json`, `practical_scale.json` and `sector_gains.json` under `runs/<study>/analyses/`.
- models and datasets: `runs/report/tables/models.csv` and `datasets.csv`.
