# the comparison models

every model solves one task: each valid turbine's relative power ([method.md](method.md)) from the same 10-minute interval's farm direction, reference speed, operating set and layout, never from the turbine's own power. the comparison set spans what a model may know about the farm. identity models learn terms for each turbine and direction or take the turbine as an input; wind-frame models describe how turbines interact by their distances along and across the wind, with no absolute direction; the absolute graph places turbines in the farm's fixed frame.

| model | name in the code | kind | what it uses |
|---|---|---|---|
| no wake | `no_wake` | reference | turbine offsets |
| power curve | `power_curve` | identity | offsets and a cell per turbine and speed bin |
| direction lookup | `direction_lookup` | identity | mean target within 5 degrees per turbine and speed bin, else the bin mean |
| cyclic GAM | `cyclic_gam` | identity | direction harmonics to order 4 per turbine and speed bin |
| identity linear | `identity_linear` | identity | offsets and 24 direction splines per turbine and speed bin |
| identity GBM* | `identity_gbm` | identity | turbine, direction and reference speed |
| absolute graph* | `absolute_graph` | absolute | absolute bearings on edges and the farm direction |
| wind-frame linear | `kernel_linear` | wind frame | offsets and windfold's shared correction |
| wind-frame GBM* | `geometry_gbm` | wind frame | along- and across-wind distances of the three nearest upstream turbines, their count and the speed |
| wind-frame graph* | `wind_frame_graph` | wind frame | along- and across-wind distances on edges |
| pretrained wind-frame graph | `pretrained_graph` | wind frame | pretrained on PyWake outputs at every degree, held-out directions included |
| PyWake* | `pywake` | physics | offsets and calibrated physics at unit weight |
| PyWake with discrepancy* | `pywake_discrepancy` | hybrid | weighted physics and a harmonic discrepancy per turbine and speed bin |
| physics-guided GBM* | `physics_guided_gbm` | hybrid | blockage ratio and distance, physics efficiency and speed |
| peer AutoML | `peer_automl` | peer | other turbines' log power, direction, speed and turbine |
| windfold | `windfold` | hybrid | offsets, weighted physics at measured inflow, the shared correction and local cells |

models marked * also run at measured inflow, under the same name with `_inflow` appended: PyWake and its discrepancy model take φ at the measured turbulence ratio, and the GBMs and graph networks get its logarithm as an input. that gives 22 comparison models besides windfold, every one scored on the same rows. peer AutoML reads same-time peer power, which already carries the wake and its inflow, and so sits at its own information level. the pretrained wind-frame graph was scored on held-out sectors only.

the Gaussian models share windfold's engine and fit their prior scales by maximizing the marginal likelihood, from three starting values. the power curve's speed cells have a zero-mean prior with a fitted variance. the direction harmonics of the cyclic GAM and of PyWake with discrepancy get precision h⁴ over a fitted variance for harmonic h, so higher harmonics shrink harder. identity linear's 24 periodic cubic splines over direction get a penalty on second differences around the circle plus a ridge. PyWake keeps the physics weight at one, and PyWake with discrepancy learns it with the prior N(1, 0.25).

three matched pairs compare descriptions of the turbines: the identity and wind-frame linear models, the identity and wind-frame GBMs and the absolute and wind-frame graphs. within a pair the two models share target, rows, loss, learner, configuration menu, tuning budget and seeds, though each pair's inputs differ in more than their frame, and every learner fits the timestamp-centered loss the engine fits. on the three farms with a calibration window, each GBM (LightGBM) and each graph network chose among 30 seeded configurations on that window and kept its choice for every split; SMARTEOLE's learners take the configurations that ranked best on the other farms. the combined GBM, one of the tests set before the confirmation period was scored, takes turbine identity, the wind-frame GBM's features and the reference speed, with the wind-frame GBM's configuration and no tuning of its own. the comparison is between these specified pipelines, not between each family at its best on held-out directions.

the sections below say, for each literature-style model, what it keeps from its source, what it changes and why. sources were read in full text, and section and page numbers point to the passages used.

## peer AutoML: the Hill of Towie Power Prediction Challenge (RES and WeDoWind, Kaggle 2025)

source. the challenge predicted one turbine's 10-minute power from its neighbors' SCADA, scored by mean absolute error in kW. the winner's slides (Conor Malone, 49.21 kW, linked from the WeDoWind results post of 27 January 2026) show AutoGluon `TabularPredictor(problem_type="regression", eval_metric="mae").fit(presets="best", time_limit=12 hours)` on a GPU, with lags of 30 to 60 minutes of four neighbors' power and speed, date features and trigonometric yaw and nacelle-position features. the second-place slides (Ahmad Chokhachian) blend AutoGluon, FLAML XGBoost and a neural network. AutoGluon's `best` is `best_quality`, with the same settings in 1.3.0, 1.4.0 and 1.6.3 and the same zeroshot model portfolio in 1.4.0 and 1.6.3 (the portfolio files compared as data); the winner's version is not stated.

kept
- AutoGluon 1.6.3 `best_quality` with its zeroshot portfolio, bagging, one stack level and weighted ensembles, as `best_quality` chooses for these data sizes.
- the other turbines' same-time data as features.

changed, and why
- same-time features only: every other valid turbine's log power, the sine and cosine of farm direction, the reference speed and the turbine ID. no lags or date features, because the task is estimation at one time, not forecasting, and no target sensors, so the inputs stay isolated from the target.
- the target is each turbine's leave-one-out peer contrast of log power, with squared loss and mean squared error as AutoGluon's metric. predictions are scaled by (m − 1)/m onto the centered scale and never centered across turbines, because each peer's prediction used the target's power.
- one model for all turbines in long format, since every turbine is a target here.
- bagging folds group whole UTC calendar days, so the rows of one day never straddle two folds. days are dealt to folds at random, with no embargo between them, so neighboring days can sit in different folds.
- dynamic stacking is off, because under grouped folds its holdout is a random row split that would leak within days. bag holdout is off, because with grouped folds AutoGluon needs separate tuning data for it.
- random forests and extra trees are left out: under grouped folds they lose their out-of-bag shortcut and overran the time budget.
- only the two neural-network families, NN_TORCH and FASTAI, get a GPU when one is usable. in 1.6.3 they default to CPU, and a GPU granted to every model would push LightGBM onto a build without GPU support, XGBoost onto a CPU-only wheel and CatBoost onto its experimental GPU mode.
- a time budget chosen per farm on its calibration window between two candidates, the longer only if it lowers validation error by more than 2%, far below the winner's. within it only the first models in the portfolio's order finish, and every fit records its leaderboard and completed models.

## absolute graph and wind-frame graph: Daenens et al. 2025

source. Daenens, Verstraeten, Daems, Nowé and Helsen, Wind Energy Science 10, 1137-1152, 2025: turbines as nodes; GENConv layers (Li et al. 2020) with softmax aggregation, message ReLU(h_j + e_ji) + 1e-7 (section 3.3); two-layer MLP encoders for nodes and edges; a dense decoder with a sigmoid; node inputs the turbine's own nacelle speed, the sine and cosine of its own nacelle direction and a turbulence intensity from a 10-minute window centered on the sample; edges between turbines in line of sight, with length and the sine and cosine of their direction; 30-second data; hyperparameters by Optuna (table 3: encoder and hidden width 64, 8 layers, dropouts under 0.1, 100 epochs, batch 64, learning rate 0.00217 with decay 0.998 for the spatial model). the optimizer is not stated and the code is not public.

kept
- GENConv with softmax aggregation and the same message and update.
- node and edge encoders of two layers, hidden width 64 and dropout in the encoders and between message-passing layers.
- table 3's ranges for the dropouts, [0, 0.1], and for the learning-rate decay, [0.9, 1]. the menu's depths are 2, 3, 4 and 8 message-passing layers, where table 3 offers 1, 2, 4 and 8.

changed, and why
- a linear output head instead of the sigmoid, because relative power is signed.
- node inputs are the reference speed (and, for the absolute graph, the sine and cosine of farm direction), never the turbine's own anemometer or vane. there is no turbulence window, because the centered window reads 5 minutes of future data. the graphs at measured inflow restore a turbulence input without future data or the turbine's own anemometer: every node carries the log of the timestamp's free-stream turbulence ratio, from the same 10-minute interval as the target.
- edges join all valid turbines within 30 rotor diameters, without line-of-sight pruning, so both frames share one edge set and wakes behind a nearer turbine still reach the network. absolute edges keep distance and the bearing's sine and cosine. wind-frame edges carry along and across distances and nodes get no absolute direction, so turning layout and wind together changes nothing (tested).
- each sample is one timestamp, and outputs are centered over its valid turbines inside the loss, the loss of the matched pairs.
- AdamW (weight decay 0.01), FP32, deterministic algorithms, dense padded tensors instead of PyTorch Geometric scatter and LayerNorm inside GENConv's MLP where PyTorch Geometric uses BatchNorm, because padded slots would enter batch statistics.
- 10-minute data, and a fixed 30-configuration menu searched on each farm's calibration window instead of Optuna on a validation year. the learning-rate range is narrowed from 1e-5 to 0.5 down to the range around both tuned optima, because a fixed random menu cannot abandon diverging rates the way Optuna does.
- a fixed optimization budget instead of epochs (item 43 of the [study plan](study_plan.md)): a network trains for a tuned number of rounds, 20 to 100, of 100 optimizer steps each, with batches drawn from a seeded reshuffle of the training timestamps whenever a pass ends and the learning rate decaying once per round. an epoch grows with the training set, so a configuration tuned on a calibration window (13,352 timestamps at Kelmarsh) would take over ten times more steps on a 167,000-timestamp random split than it was tuned with; with rounds, tuning and every split take the same steps. on the Kelmarsh calibration split, menu configuration 0 scored a validation MSE of 0.083 after its 35 epochs (490 steps) and 0.036 after its 35 rounds (3,500 steps).
- batches of 256 to 2048 timestamps instead of 64 to 512, so that a fixed number of steps sees the training set more than once: 3,500 steps at batch 64 would see a 167,000-timestamp split about 1.3 times. tuning still chooses within the range.
- the spatio-temporal variant (an LSTM over the last 5 minutes) is not used, because the task is same-time.

## physics-guided GBM: Zehtabiyan-Rezaie et al. 2023

source. Zehtabiyan-Rezaie, Iosifidis and Abkar, PRX Energy 2, 013009, 2023 (section II.B.1, equations 3 to 5, table II): XGBoost predicts each turbine's efficiency from its blockage ratio (the share of its rotor disc inside the cylindrical wakes of upstream turbines), its cumulative inverse blocking distance (over upstream turbines, the disc share each one blocks weighted by one over the distance) and an efficiency from the Park or Niayifar and Porté-Agel model; labels come from RANS simulations.

kept
- both geometric features as defined, with cylindrical wakes one rotor diameter wide behind every upstream turbine and the cumulative form of the inverse distance, plus a physics efficiency, in a boosted tree ensemble.

changed, and why
- LightGBM on the timestamp-centered loss instead of XGBoost on efficiency labels, so the hybrid and the other GBMs share learner, loss and menu.
- SCADA targets (relative power) instead of RANS efficiencies.
- the efficiency is exp(φ) from the calibrated PyWake family chosen in each split by training error, as for PyWake. the Niayifar and Porté-Agel family is the paper's NP model, and Jensen's is the wake model inside its Park model.
- only operating turbines cast wakes, and the integrals use 128 equal-area points on the disc and distances in rotor diameters. hub-height differences shift the wake vertically.
- the reference speed is a fourth feature, because SCADA rows span wind speeds.
- at measured inflow the efficiency is computed at the measured turbulence, the NP model's turbulence input, and the log turbulence ratio is a fifth feature.

## pretrained wind-frame graph: Schröder et al. 2022 and Duthé et al. 2024

sources. Schröder, Dimitrov, Verelst and Sørensen, Energies 15, 558, 2022 (sections 2.3, 2.4 and 3.5): a 3-layer, 25-unit network pretrained for 400 epochs on 30,000 HAWC2 and dynamic wake meandering simulations, then retrained on 1 to 12 months of SCADA with its first layers frozen. Duthé, de N Santos, Abdallah, Weijtjens, Devriendt and Chatzi, Data-Centric Engineering 5, 2024: a GEN-layer graph network pretrained on PyWake simulations of random layouts, fine-tuned to an unseen layout simulated in HAWC2Farm by full tuning, decoder-only tuning or LoRA; with little data LoRA and decoder-only did best, and with more data full tuning and LoRA kept improving.

kept
- pretraining on engineering-model outputs, then fine-tuning on short SCADA histories, judged by accuracy against history length (Schröder); a GEN-layer graph pretrained on PyWake with flow-relative edges (Duthé).

changed, and why
- the network is the wind-frame graph with the architecture its calibration-window tuning chose, so pretraining changes only the starting weights of a tuned model.
- pretraining data come from the farm's own layout: PyWake contrasts from one teacher, the calibrated family with the lowest calibration-window error, over a 1-degree direction grid, the speed-bin centers and the calibration window's operating sets plus all turbines operating, with seed 0, 100 rounds of 100 steps and no early stopping. SMARTEOLE, with no calibration window, uses all turbines operating and the teacher most other farms chose, with its median parameters. pretraining sees conditions and PyWake power only, never SCADA power (tested), but its direction grid includes held-out directions.
- fine-tuning updates all weights with the chosen configuration's training options (Duthé's full tuning), rather than freezing layers or LoRA, which would add options the menu does not tune.
- the targets are SCADA power only, with no loads, rather than HAWC2 or HAWC2Farm outputs.

## PyWake with discrepancy: in the spirit of Andersson et al. 2020

source. Andersson, Doekemeijer, van der Hoek, van Wingerden and Imsland, Journal of Physics: Conference Series 1618, 022043, 2020 (pages 4 and 11): an engineering wake model corrected by a zero-mean Gaussian process with an automatic-relevance squared-exponential kernel, learned from SOWFA data; far from data the correction returns to zero and the engineering model remains.

kept
- the engineering model as the mean and a zero-mean Gaussian discrepancy per turbine, so unsupported corrections shrink back to the physics.

changed, and why
- the discrepancy is a finite cyclic basis over direction (harmonics to order 4, by speed bin) with Gaussian priors and scales from the marginal likelihood, inside windfold's engine: a Gaussian process with a finite basis that scales to millions of rows and gives intervals the same way as the other engine models.
- inputs are direction and speed bin per turbine, not control inputs, and targets are SCADA relative power, not LES power.
- the physics term is the calibrated PyWake family chosen per split by training error, with a learned weight.

## direction lookup: wind-up's directional method

source. wind-up v0.7.0 (RES), `wind_up/detrend.py`: for every whole degree, the rows whose reference direction lies within half a 10-degree bin (plus or minus 5 degrees) give a ratio of mean speeds between test and reference turbine, kept only with enough hours above and below 8 m/s, per waking scenario, and interpolated over direction.

kept
- a per-turbine table on whole-degree bins with a window of plus or minus 5 degrees, the directional detail operators use.

changed, and why
- the table holds the mean of the turbine's relative power in each window, per speed bin, instead of a speed ratio to one reference turbine, because every model estimates the same quantity. it is fitted outside the Gaussian engine and has no predictive variance.
- an empty cell falls back to its turbine's speed-bin mean instead of being dropped, so the lookup predicts every test row. as a monitor it abstains where fewer than 20 training rows support a cell.
- no waking scenarios or IEC hour thresholds: operating sets enter through which rows are valid.
