# the method

windfold learns how the turbines of one farm take wind from each other, from 10-minute SCADA rows, in one Gaussian model: an offset per turbine, plus calibrated PyWake physics at the measured turbulence with a fitted weight, plus one correction shared by every turbine pair in the wind's frame, plus local cells per turbine, direction and speed bin. [baselines.md](baselines.md) describes the comparison models and [results.md](results.md) what the study found.

## what every model estimates

for turbine i at timestamp t, with m_t turbines whose rows pass the checks of the [data cards](data/), every model estimates the relative power

    y_ti = log p_ti − (1/m_t) Σ_j log p_tj

where p_ti is the turbine's mean power over the 10-minute interval. subtracting the mean removes the level all turbines share at that moment, and with it most of the weather: if every turbine produces 10% more, no y changes, and if one turbine falls into a wake, its y drops. the inputs are the farm's wind direction θ_t, a reference speed in one of three bins between cut-in and rated speed (5 to 6.5, 6.5 to 8 and 8 to 10 m/s), the set of operating turbines and the layout. the reference speed is the median nacelle speed of the free-stream turbines, and a turbine's own power is never an input.

a row is waked when an operating turbine within 30 rotor diameters stands upwind with the wind inside its disturbed sector, which for two turbines L apart is 1.3 arctan(2.5D/L + 0.15) + 10 degrees wide (IEC 61400-12-1), centered on the bearing from the downstream turbine to the upstream one. errors are reported on waked rows, where models differ and where a turbine's own anemometer never enters the reference speed. each model returns one estimate per row, and the Gaussian models a predictive variance with it.

## the model

before centering,

    log p_ti = μ_t + α_i + β φ_ti + Σ_{j ∈ U_ti} g_k(a_ij, c_ij) + v_{i,b,k} + ε_ti,    ε_ti ~ N(0, σ²_k)

- μ_t is the level all turbines share at t. centering removes it, so it is never estimated.
- α_i is the turbine's offset. differences in level between turbines, set by terrain, local wind and sensors, can persist across directions.
- φ_ti is the wake physics for this turbine at this timestamp, and β its weight.
- g_k is the shared correction for speed bin k: a smooth function of where an upstream turbine j sits, a_ij along and c_ij across the wind from turbine i in rotor diameters, summed over the operating upstream turbines U_ti inside its window.
- v_{i,b,k} is the local cell of turbine i, 5-degree direction bucket b and speed bin k. it can absorb terrain or sensor effects the shared terms cannot.
- σ²_k is a noise variance for each speed bin.

the terms can trade off against each other, so the correction is a learned adjustment, not an identified physical wake.

## physics

φ_ti is the log of turbine i's PyWake power, with only the turbines operating at t casting wakes, minus the log of its free-stream power, both floored at 1% of rated power. PyWake 2.6.20 evaluates it at the center of the speed bin, the whole degree nearest the farm direction and each turbine's own hub height, with power and thrust curves from the turbine configs, each number with its source. three wake model families are used:

- Jensen's model, with wake decay constant k_w;
- the Gaussian model of Niayifar and Porté-Agel, with ambient turbulence intensity I₀;
- TurbOPark, with expansion A and I₀. with A and I₀ together, I₀ is only weakly identified and can sit at its search bound.

each family also averages power over neighboring whole degrees with Gaussian weights of a fitted spread, which stands in for the scatter of direction within 10 minutes and between turbines. rows share far fewer (operating set, whole degree, speed bin) keys than there are rows, 25,587 keys for Hill of Towie's 768,071 development rows, so PyWake runs once per key and the values are joined back to the rows.

each family is calibrated on the farm's calibration window, as FLASC calibrates wake models on SCADA: Optuna's tree-structured Parzen estimator runs 100 seeded trials over the family's parameters and spread, minimizing the error of turbine offsets plus φ at unit weight and constant inflow. each split then uses the family with the lowest such error on its own training rows. SMARTEOLE, which has no calibration window, takes each family's median parameters over the other farms.

## measured inflow

with the same direction and speed, a calm night gives deeper and longer wakes than a gusty afternoon, because the ambient turbulence sets how fast a wake mixes out and how far it wanders. a φ computed at one turbulence level for every timestamp leaves that variation in the residual. windfold therefore scales each family's calibrated, turbulence-dependent parameter (k_w for Jensen's model, I₀ for the other two) by the timestamp's turbulence ratio

    r_t = I_t / I_ref,    k_w,t = k̂_w r_t,    I₀,t = Î₀ r_t

where I_t is the free-stream turbulence intensity of the interval, the median over free-stream turbines of the within-interval standard deviation of nacelle speed over its mean, and I_ref its median over the calibration window (for SMARTEOLE, the median of the other farms' references). this adapts the proportional dependence of wake decay on turbulence of Duc et al. to one shared free-stream measurement, and the calibrated parameters are not refitted. φ is computed on a grid of ratios from 2^−1.5 to 2^1.5 in steps of 2^0.25 and interpolated linearly in the log ratio, so a ratio of one gives the constant-inflow φ exactly. ratios outside the grid are clipped, which touches under 0.05% of the timestamps of any period. a timestamp without a measurement takes a ratio of one; this happens only at Kelmarsh, on 47.5% of its calibration timestamps and 7 development timestamps.

I_t also tracks the scatter of direction within the interval: at 6 to 9 m/s its Spearman correlation with the median direction standard deviation over the unwaked turbines is 0.81 at Kelmarsh and 0.86 at Penmanshiel. the ratio therefore stands for mixing and meandering together. windfold, PyWake at measured inflow and PyWake with discrepancy at measured inflow read φ at the measured ratio, and the GBMs and graph networks at measured inflow get log r_t as an input, so matched models see the same measurement.

## the shared correction

for a farm direction θ, every pair of turbines has a distance from source to receiver along the wind and across it, in rotor diameters. a turbine counts as upstream inside the window 0 < a ≤ 30 and |c| ≤ 3. the pair (j, i) at θ has the same relation as the pair (i, j) at θ + 180 degrees, and because the correction is shared, a relation learned from one pair in one direction applies to every pair with that relation in any direction.

<p align="center"><img src="figures/idea.png" width="100%" alt="why the wind's frame carries into unseen directions, from Penmanshiel's records over the 30 days from 1 April 2017. (a) turbine and direction cells with training rows: directions near 40 to 100 degrees have none. (b) the same rows as relations along and across the wind, in rotor diameters, with the relations the unseen directions need marked as points, almost all inside cells already seen. (c) for each direction, the share of waked turbines with training rows of their own falls to zero in the unseen band, while the share of their upstream relations seen in training stays at 100%"></p>

in 30 days of Penmanshiel's records a band of directions has no rows, so a model that learns each turbine's pattern by direction has nothing there, yet in the wind's frame the same rows already cover nearly every relation those directions need. seen cells measure geometric coverage, not whether a wake effect can be estimated there.

for each speed bin, g is a cubic tensor B-spline with 12 × 11 = 132 basis functions on a fixed grid of knots, dense near the rotor where wakes change fastest: along the wind at 0 and at nine points from 1.5 to 30 diameters, across it symmetrically at four points from 0.45 diameters outward. its coefficients w have the prior N(0, λ Q⁻¹), where Q penalizes second differences along and across the wind, first differences between speed bins and, with weight 0.1, the coefficients themselves. g is thus a Gaussian process over pair relations and speed bins. a row's design columns are the sum of the basis over its operating upstream turbines, so they depend only on directions, speed bins, operating sets and the layout. they are built once per farm and stored under `kernel_design/` with hashes of the rows and basis they came from, and a fit refuses a design built for other rows, knots or window.

## local cells

the local cells are many (21 turbines × 72 buckets × 3 speed bins at Hill of Towie, 4,536 coefficients) but simple in shape. all rows of a timestamp share one bucket and one speed bin, so the data couple local coefficients only within a cell, across its turbines. each cell's data matrix is turned into its eigenvectors once; in that basis the cell's precision is diagonal for any noise and τ², and a Schur complement removes all local cells from each evaluation of the marginal likelihood. a local cell without training rows keeps its prior, so in an unseen sector the estimate leans on the offsets, the physics and the shared correction.

## fitting

the priors are α_i ~ N(0, 1) and β ~ N(1, 0.25), with variances as the second argument. the prior scales λ of the shared correction and τ² of the local cells are fixed at 0.04 rather than fitted, a choice made on the development window that mattered mainly on SMARTEOLE, the farm with the fewest timestamps, where maximizing the marginal likelihood over a few thousand timestamps overfits them.

centering leaves m_t − 1 independent contrasts per timestamp, not m_t. projecting the rows with H = I − 11ᵀ/m_t gives exactly the Gaussian likelihood of those contrasts, so the fit accumulates centered sums per speed bin once (XᵀHX, XᵀHy, yᵀHy and the number of contrasts, with each local cell's own blocks) and never touches the rows again. the coefficients integrate out in closed form, and the noise variances maximize the resulting marginal likelihood, by L-BFGS-B over their logarithms with an analytic gradient. everything is float64, and posteriors are conditional on the fitted scales.

## prediction and uncertainty

a prediction is the posterior mean, with variance

    Var(y_ti | data) = x̃_tiᵀ V x̃_ti + σ²_k (1 − 1/m_t)

where x̃_ti is the centered design row and V the posterior covariance of the coefficients; a local cell without training rows adds its prior variance τ², passed through the centering. the same machinery gives the variance of any linear combination of predictions, such as one turbine's sum over a day, with every cross-covariance kept and without forming a rows × rows matrix.

because m_t y_ti / (m_t − 1) is the log ratio of a turbine's power to the geometric mean power of the other valid turbines, an estimate converts to power given the other turbines' observed power, and errors in kW use this conversion. sums of power carry Duan's smearing factor for the log transform.

## windfold's variants

twelve variants isolate windfold's parts. their names are those of the output folders.

| variant | what changes |
|---|---|
| `windfold_no_physics` | no physics term |
| `windfold_no_kernel` | no shared correction |
| `windfold_no_local` | no local cells |
| `windfold_permuted_layout` | turbine positions permuted in the correction's relations by a seeded derangement, the same for every split |
| `windfold_default_physics` | PyWake 2.6.20's published default wake parameters instead of calibrated ones |
| `windfold_unit_physics` | physics weight held at one |
| `windfold_constant_inflow` | constant inflow, r_t = 1 |
| `windfold_evidence_scales` | prior scales fitted by maximizing the marginal likelihood |
| `windfold_constant_inflow_evidence_scales` | both of the last two: the first configuration |
| `windfold_average_inflow` | φ replaced by its average over the ratios measured at every other timestamp of the period |
| `windfold_climatology_inflow` | the same average over timestamps with the same 30-degree sector, speed bin and month |
| `windfold_diurnal_inflow` | the same average also matched on the 6-hour part of the day in UTC: the diurnal control |

the three inflow controls ask whether the measured turbulence matters beyond what is expected for the conditions. averaging φ rather than the ratio keeps the nonlinear wake response, and leaving the timestamp out of its own average keeps its measurement out. a group with one timestamp takes its period's average. the averages can include turbulence measured in a held-out sector, never its power. each control refits the whole model, the condition-and-relearn design of Hooker et al. (2021) with nested conditioning sets (König et al. 2021).

the tighter variant tested on the confirmation period (`windfold_restricted`) is windfold without local cells and with the shared correction's prior variance at 0.04/64. wind-frame linear, offsets plus the shared correction alone, is windfold's representation without physics ([baselines.md](baselines.md)).

## the coverage score

the third planned test asks whether a score computed from the design alone predicts where wind-frame linear fails on a held-out sector. for each upstream turbine j of a waked test row, the score takes the share of the shared correction's prior variance at that pair's relation that remains after the correction alone, with fixed scales, is fitted to the training rows. it weights that share by φ₁(r_ij)², the squared effect of j's wake on i as if the two stood alone, under PyWake's Niayifar and Porté-Agel model at its default parameters, and sums over j:

    score = Σ_j φ₁(r_ij)² · var_post[g(r_ij)] / var_prior[g(r_ij)]

a sector's score is the mean over its waked rows. at fixed scales the posterior covariance of a Gaussian model depends only on the design, so the score reads directions, speeds and operating sets but no power, and a test permutes the power values and finds it unchanged. the operating sets themselves come from screening that reads power. the score is compared with four simpler predictors: the angular distance to the nearest training direction, the expected wake strength (the sum of φ₁² over upstream turbines), the number of training rows and the dissimilarity index of Meyer and Pebesma (2021) on the wind-frame features, without their variable-importance weights. a permuted-layout null computes the same score with the turbine positions permuted in the correction's relations alone and is reported beside the test.

## checks

`uv run pytest` runs the tests behind these claims:

- the projected likelihood equals the likelihood of an explicit orthonormal basis of contrasts, the posterior equals a dense solve, the Schur path equals the dense path, the analytic gradient equals finite differences and a day's variance with its cross-covariances equals the dense answer;
- the cached φ equals a direct PyWake call on the same flow cases, calibration recovers wake parameters from data generated with known ones and the physics does not depend on the number of worker processes;
- the stored correction design equals independent spline sums, and turning the wind by half a turn swaps the roles of every pair;
- on a synthetic six-turbine layout, wind-frame linear recovers a known wake-like correction where training rows reached it: with a correlation above 0.95, an error under a tenth of its depth and 90% intervals covering 80 to 100% of those grid points. its 90% predictive intervals lie within 3 points of nominal on held-out timestamps.
