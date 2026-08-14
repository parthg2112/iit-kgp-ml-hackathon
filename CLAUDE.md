# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment — read this first

The `python` on PATH is **3.13 and has no scientific stack**. Everything must run under 3.11:

```
C:/Users/USER/AppData/Local/Programs/Python/Python311/python.exe
```

numpy 2.4.6 · scipy 1.17.1 · scikit-learn 1.7.0 · pandas 2.3.0. There is no venv and no
`requirements.txt`; the 3.11 interpreter already has what is needed.

## Commands

```bash
PY="C:/Users/USER/AppData/Local/Programs/Python/Python311/python.exe"

$PY scripts/check_integrator.py    # gate: splitting scheme vs scipy BDF, must PASS
$PY scripts/run_baseline_cv.py     # repeated 10-fold CV, all tree variants (~3 min)
$PY scripts/fit_physics.py         # single-model ODE fit; --maxiter/--popsize/--seed/--tag
$PY scripts/compare_models.py      # fit every variant in physics.MODELS and rank them
$PY scripts/run_physics_cv.py      # honest CV, params refit per fold + blend search (~10 min)
$PY scripts/make_submission.py     # end-to-end -> "Claude ke Chatore.csv"
$PY scripts/build_notebook.py      # regenerate notebook/final.ipynb
$PY scripts/make_figures.py        # figures/physics_diagnostics.png

# Diagnostics — run these before adding any parameter to the model
$PY scripts/diagnose_fit.py        # self-recovery + poison-hit count (see below)
$PY scripts/check_bounds.py        # refit with the box widened; is the optimum pinned?
$PY scripts/check_cold_folds.py    # cold-start folds; the honest generalization number
$PY scripts/sweep_tanks.py         # tanks-in-series: does plug flow hold?
$PY scripts/profile_params.py --param E2_kJ   # profile likelihood; also a1, n_flow, E1_kJ
$PY scripts/verify_residual_hybrid.py         # does a residual corrector earn its place?
$PY scripts/tree_sign_test.py                 # bias or variance? band-by-band; names its stratum
$PY scripts/weighted_ensemble.py              # is there a posterior worth averaging over?
$PY scripts/audit_rows.py                     # row-level verdicts vs an external audit
```

Scripts insert the repo root on `sys.path` and import `src.*`; run them from the repo root.
Output lands in `artifacts/`. Run long fits in the background with `-u` and redirect to
`artifacts/<name>.log` — without `-u`, piping to `tail` buffers everything to the end.

Re-execute the notebook after any result changes; it reads `artifacts/` at runtime:
`$PY -m nbconvert --to notebook --execute --inplace notebook/final.ipynb`

**Budget note:** `least_squares` uses a finite-difference Jacobian, so each `nfev` costs
`n_free + 1` integrations. `max_nfev=6000` made a single model take ~25 minutes for no
accuracy gain; 800 is plenty. Polish converges long before the cap.

**Never pass a lambda or closure to `differential_evolution`.** It cannot be pickled, so
`workers=-1` is impossible and the fit silently runs on **one of 12 cores** — this cost
~10x on `check_bounds.py` (25 min) and `profile_params.py` (8.7 min/point) before it was
caught. Use the module-level `pinned_objective` / `pinned_residuals` in `src/physics.py`,
which take the pinned parameters via `args=(pin_names, pin_values, ...)`.

For a grid of *independent* fits, put the parallelism one level up — a process per grid
point via `multiprocessing.Pool` with each fit single-threaded — rather than `workers=-1`
inside each fit. `profile_params.py` does this: 10 points went from ~70 min to ~9 min.
Do not do both at once; that oversubscribes the cores.

This workload is **not** GPU-suited and no GPU path is worth building: the integrator is a
sequential loop of 512–2048 steps over 150-element arrays, so kernel-launch overhead would
exceed the arithmetic. Parallelism across fits is the only lever.

## The task

Surrogate model for a slow non-isothermal plug-flow reactor simulation. 150 train rows,
50 test rows, 5 inputs → `overall_yield` (0–100). Scored blind on RMSE. Top 6 advance to a
live pitch judged on **process insight**, not algorithm choice — Phase 2 is half the score,
so physical interpretability is a deliverable, not a nicety.

**One submission ever. No leaderboard feedback.** Every decision is made against
cross-validation, never against a probe score.

### Submission contract (problem statement §5)

- File named **`claude_ke_chatore.csv`** — team name is "Claude ke Chatore", but the file is
  written with underscores (no spaces) for the upload platform. Do not "fix" this back to
  the spaced form; it was a deliberate call.
- Exactly 50 rows, in the **same order** as `test_dataset.csv`
- Exactly **one** column, header `overall_yield` — no index column, no ID column
- Floats to ≥3 decimals, all within `[0, 100]`
- Finalists must also submit the documented `.ipynb`

`src.data.write_submission` enforces all of this and re-reads the file to re-validate.
Do not hand-write the CSV.

## Layout

`README.md` is the teammate-facing orientation; this file is the engineering detail.

```
claude_ke_chatore.csv   THE SUBMISSION — never hand-edit; rebuild via make_submission.py
notebook/final.ipynb    finalist deliverable; SELF-CONTAINED, no src imports
data/                   train_dataset.csv, test_dataset.csv  (paths in src/data.py)
docs/                   problem_statement.pdf, guide.md (team brief — not ground truth)
reference/              an external audit's competing predictions, kept for comparison
artifacts/ figures/     fitted params, CV results, profiles, plots
```

**The notebook must stay self-contained.** Finalists submit the `.ipynb`, and one that
imports from `src/` will not execute for a judge. `build_notebook.py` inlines the real
function source via `inspect.getsource` (see its `embed()` helper), so the notebook cannot
drift from the tested code. Do not "simplify" it back to imports.

## Architecture

Two independent models plus a searched, regime-aware blend:

```
src/data.py      loading, physics feature engineering, submission writer + validator
src/physics.py   the reactor ODE model — 7 parameters, custom integrator      <- the main event
src/baseline.py  ExtraTrees safety net on physics features
src/residual.py  hybrid: physics prediction + regularized correction on its residuals
src/evaluate.py  repeated K-fold harness, per-fold physics refit, blend-weight search
```

### The reactor model (`src/physics.py`)

`A --k1--> B --k2--> C`; B is the product and is being made and destroyed at once.

```
dCA/dz = -k1*CA
dCB/dz =  k1*CA - k2*CB
dT/dz  =  a1*k1*CA + a2*k2*CB + U*(T_jacket - T)
```

Seven fitted parameters: `ln_k1_ref, E1_kJ, ln_k2_ref, E2_kJ, a1, a2, U`.

Three non-obvious implementation decisions, each of which was load-bearing:

1. **Arrhenius is reparameterized about `T_REF = 430 K`** (`k = exp(ln_k_ref)·exp(-E/R·(1/T - 1/T_ref))`).
   Fitting `ln A` and `E` directly correlates them >0.999 and turns the objective into a
   long narrow valley. This is why the fit is *not* slow to converge.
2. **All 150 rows integrate simultaneously** as numpy arrays over a normalized axis, with
   an operator-splitting scheme: rates frozen per substep, mass balance advanced
   *analytically* (exact solution of the linear series reaction), energy balance advanced
   with the exact linear-ODE solution. Both halves are unconditionally stable, so no step
   size produces negative concentrations or oscillation. A midpoint predictor–corrector
   makes it second order. One 150-row evaluation is ~22 ms at 512 substeps.
   **Do not replace this with fixed-step RK4** — RK4 blew up on thermal-runaway rows.
3. **`residuals()` can poison parameter sets whose integration has not converged**, by
   re-integrating at half resolution and rejecting on disagreement > 0.05. Without it the
   optimizer minimizes *integration error* instead of data error: an early fit produced
   parameters whose predictions moved 92 yield-points when substeps were increased.
   The `a1`/`a2` bounds of ±30 K·L/mol exist for the same reason (caps each reaction's
   adiabatic excursion at ~120 K given CA0 ≤ 3.97).

   **The guard defaults OFF and must be passed explicitly (`convergence_tol=CONVERGENCE_TOL`)
   — do that only at differential-evolution call sites.** It is a cliff in an otherwise
   smooth objective, so a `least_squares` finite-difference Jacobian straddling it is
   garbage. It fires 159/256 times around the optimum; leaving it on during polish costs
   4.38 instead of 3.66 and wrecked 18 of 24 polish starts.

`reference_solve()` is the independent scipy BDF check — keep `check_integrator.py` passing
after any change to the integrator.

### Feature engineering (`src/data.py`)

Derived features are reaction-engineering groups, not combinatorics: `tau = length/flow`,
`log_tau`, `T_avg`, `delta_T`, `inv_T_avg`, and `ln_Da_E = log(tau) - E/(R·T_avg)` for
E ∈ {60, 100, 160} kJ/mol. Add features in that spirit or not at all.

Note `tau = L/Q` has units m·min/L, not time — true `tau = A_c·L/Q`. Tube diameter is
constant across rows, so `A_c` folds into the fitted pre-exponentials and into `U`.
Harmless, but state it rather than let a judge find it.

## Measured facts (verified here, not inherited)

| Quantity | Value |
|---|---|
| Exact zeros in target | 24.7% (38% below 1.0) — zero-inflated and bimodal |
| corr(`jacket_temperature_K`, yield) | −0.498 |
| corr(`concentration_mol_L`, yield) | +0.009 |
| corr(`log_tau`, yield) | +0.061 |
**Quote the protocol with every number — they are not comparable.** `train` = fit and scored
on all 150. `10f-CV` = repeated 10-fold, parameters refit per fold, warm-started. `cold` =
folds refit by differential evolution from scratch. `LOSO` = policy hyper-parameters chosen
on two seeds and scored on the third (the only figure that prices hyper-parameter selection).

| Quantity | Protocol | Value |
|---|---|---|
| ExtraTrees + physics features | 10f-CV | 16.37 ± 1.38 |
| Physics ODE, raw | train @2048 substeps | 3.6559 |
| Physics ODE, raw | train @512 substeps | 3.6617 |
| Physics ODE, raw | 10f-CV | 6.358 ± 0.124 |
| Physics ODE, raw | cold | 2.97 / 13.05 / 1.93 (mean 5.98) |
| Physics ODE, noise-averaged σ=1.67 | 10f-CV | 5.935 |
| Previous policy (raw, w=0.910, c=60) | LOSO | 6.022 |
| **SHIPPED: σ=1.67, w=0.87, cutoff=60** | **LOSO** | **5.671** |

A 50-row test set adds large sampling noise on top of any of these: bootstrapping the OOF
predictions gives a 5th–95th percentile of **[2.41, 9.07]**.

The warm-started CV and the cold folds agree at ~6.0, which is the honest generalization
estimate. An earlier run reported 3.662 ± 0.000; that was a bug (see the guard note above),
not a result.

**The `± 0.000` was an artifact, not a stability result — and it had two causes.**
`make_physics_predict_fn` warm-starts each fold's `least_squares` from the full-data
optimum, so on 135 of 150 rows the "refit" barely moves. Worse, that early run had the
convergence guard on by default, and the guard fires 159/256 times in exactly that
neighbourhood — so the folds hit the poison wall immediately and could not move at all.
Not near-zero movement: zero. A zero seed-to-seed spread means the folds were not
independent of the rows they were scored on.
`check_cold_folds.py` refits with differential evolution from scratch and is the honest
protocol. Use it before quoting any generalization number.

Fold 1's poor held-out result is **genuine, not a search failure**: re-run at ~3x the global
search budget it reproduced train 2.5618 / held-out 13.0530 to four decimals. On 135 rows a
distinct parameter basin fits better but generalizes worse; on all 150 it is no longer
competitive. E1 and ln_k1 shift only ~4% across folds, but a2 shifts 91% and ln_k2 61% —
the *desired* reaction is tightly determined, the waste reaction is not, because k2 is
either negligible or overwhelming across most of the sampled conditions. The governing
ratio k2/k1 stays stable (0.063–0.079) regardless.

Note `check_cold_folds.py` overwrites `artifacts/cold_folds.json` with however many folds
it was asked for — re-run with the same `--folds` before regenerating the notebook.

Final model: `series` (7 free parameters, `n_flow` pinned to 0), train RMSE **3.6617 at 512
substeps / 3.6559 at 2048** — always state which, they differ in the third decimal and both
appear in the artifacts. Recovered: E1 = 43.16, E2 = 250.07 kJ/mol, a1 = −11.79,
a2 = +11.34 K·L/mol, U = 3.2552. The k1 = k2 crossover sits at **449.9 K**, inside the
training temperature envelope of 351.6–548.0 K (`scripts/pitch_evidence.py`).
`check_bounds.py` refit with the box widened to E1∈[5,400], E2∈[20,600] and landed on the
same values with nothing at a constraint — the activation energies are real, not artifacts.

## The shipped prediction policy: (sigma, weight, cutoff) = (1.67 K, 0.87, 60)

All three are chosen **jointly** (`scripts/joint_policy.py`) — optimising down a chain lands
wherever the ordering happens to put you. Leave-one-seed-out on the whole triple:
**6.022 → 5.671, better on all three held-out seeds.**

**σ is fixed a priori at 1.67 K**, the errors-in-variables median from `scripts/eiv_noise.py`
— *not* tuned on the CV that then judges it. That matters: the in-sample argmax (σ = 2.0)
scores 5.640, but quoting that would be reporting a gain measured at its own optimum. Only
the physics component is smoothed; the tree is fit on observed inputs and is already
implicitly averaged over them, so blurring it too would double-count the noise.

**A single global weight hid a defect.** By prediction stratum the tree *helps* mid-range and
*hurts* at both ends — worst on the near-zero rows the physics gets almost exactly right, and
again at the top where a tree cannot extrapolate. Hence the **regime-aware** rule: blend below
a predicted yield of 60, pure physics above (`BLEND_CUTOFF` / `apply_blend` in
`src/evaluate.py`). The cutoff came out at 60 in every leave-one-seed-out fold.

**The tree's 13% is variance reduction, not bias correction** — tested twice, on two
different strata, and **always quote which stratum a sign test was run on**
(`scripts/tree_sign_test.py`).

| Sign test | Stratum | Result | Status |
|---|---|---|---|
| First | mid-range 10 < p ≤ 60 | 22/42 (52%, p = 0.88) | **superseded — wrong stratum**; the mid-range contributes *none* of the gain |
| **Decisive** | **0.5 < p ≤ 10, where the entire +0.27 lives** | **8/13 (62%, p = 0.58)** | bar was ≥ 11/13; per-seed 7/12 · 9/13 · 9/14, no seed clears it |

The first test was not wrong in method, only in stratum choice — it measured the band where
the tree is wrong-signed rather than the band that carries the gain. Re-run where it matters,
the verdict is unchanged: no significant sign alignment, so **variance reduction, not bias
correction**. What the tree does have is decorrelated error (correlation with the physics
error **+0.070**), and a weak decorrelated component improves an ensemble by averaging even
when far worse standalone (15.0 vs 6.1 RMSE on the blended rows).

*(Beware `corr(tree − physics, y − physics)`: both share `− physics`, so it is spuriously
high. The sign test is the clean statistic — but only on a stratum where `need` and `pull`
both vary in sign; see the dead-band note below for where it degenerates.)*

**The blend buys variance by paying bias — quantified.** Pure smoothed physics is essentially
unbiased on the blended rows (mean signed OOF residual **+0.066**); blending shifts that to
**+1.148**, an introduced bias of **+1.083** (exactly `(1−w)·mean_pull`). Restricted to those
blended rows, RMSE still improves 6.142 → 5.807 — a subset figure, quoted here only to price
the bias; the whole-set headline is the per-seed **5.9352 → 5.6541**. The trade is worth
taking, but it is a weaker footing than bias correction would have been and should be stated
that way.

Per-band breakdown (seed-averaged predictions). **Three different numbers describe this
policy and they are not interchangeable — always name the protocol:**

| Number | Protocol | What it is |
|---|---|---|
| **5.671** | **LOSO** | **the headline: policy chosen on two seeds, scored on the third** |
| 5.9352 → 5.6541 | 10f-CV, per seed then averaged | the blend's gain with the policy held fixed |
| 6.142 → 5.807 | 10f-CV, blended rows only | the subset used to price the introduced bias |

LOSO is the only one that prices hyper-parameter selection, so it is the number that goes in
front of a judge. The per-seed pair below is *not* LOSO.

| Band | n | mean pull | applied | needed | band RMSE gain |
|---|---|---|---|---|---|
| dead, p ≤ 0.5 | 56 | +5.92 | +0.77 | +0.13 | **−1.547** |
| low, 0.5 < p ≤ 10 | 13 | +25.28 | +3.29 | +4.34 | **+1.741** |
| mid, 10 < p ≤ 60 | 29 | +5.38 | +0.70 | −2.41 | +0.232 |

**"The whole gain is 13 rows" was WRONG — do not repeat it.** Per-band *RMSE deltas are not
additive* and reading them as contributions is the error. Redone in SSE, which is additive
(`scripts/blend_accounting.py`, pooled over 3 seeds × 150 rows = 450 predictions, closure
residual 4.6e-13):

| Band | n | RMSE pre → post | ΔSSE | % of gain |
|---|---|---|---|---|
| dead, p ≤ 0.5 | 168 | 0.366 → 1.960 | **−622.9** | **−42.5%** |
| low, 0.5 < p ≤ 10 | 39 | 11.210 → 9.580 | +1321.3 | +90.2% |
| mid, 10 < p ≤ 60 | 86 | 9.441 → 8.957 | +765.9 | **+52.3%** |
| above cutoff | 157 | unchanged | 0.0 | 0.0% |
| **TOTAL** | **450** | **5.935 → 5.655** | **+1464.3** | **100%** |

The mid band contributes **half the gain**, despite a small RMSE delta, because it holds 86
rows at a baseline RMSE of 9.44. And the low band's gain is **tail-dominated**: the top 3 of
39 rows supply **77%** of it, with pre-blend errors of 21–38 yield-points against a band
median |need| of only 1.46.

**State the dead-band cost as a level, not just a delta:** slice RMSE **0.366 → 1.960**
(delta −1.594, cost 622.9 SSE). "Loses 1.55 RMSE" is ambiguous between the two.

**Do not quote the dead band's sign test (17/56, p = 0.005) as a finding — it is
near-tautological.** The tree's pull is positive on **56 of 56** dead rows (a tree cannot
predict below its training floor), so the sign test there collapses to "is `y > phys`",
which is just the 39/56 rows where the physics is already at or above truth. The
non-tautological statement is the RMSE cost, **−1.547**. Lead with that.

**A lower gate was measured and NOT adopted.** Blending only where p > 0.5 scores 5.5309 vs
the shipped 5.6541 — but that is in-sample, on the rows that would judge it, and it is a
fourth free knob on a mechanism already rejected once as mechanism-plus-patch. Reported for
honesty, not taken. `scripts/tree_sign_test.py` prints the sweep.

The weight is on a **broad plateau**, not a peak: gain 0.281 / 0.279 / 0.271 at w = 0.87 /
0.85 / 0.89, falling off only below 0.80. Not a tuning artifact.

**Bagged physics was tried as a replacement and failed** (`scripts/bagged_physics.py`). If
the gain is pure averaging, bootstrapping the physics fit should deliver it with one model
and no upward pull. It recovers only **+0.132 of the +0.281**, and on one seed it is *worse*
than the single fit (6.62 vs 5.87). Bagging does keep predictions unbiased (+0.072), it just
does not buy the variance reduction.

**Report the failure as a robustness finding, not a dead end.** The reason it fails is the
interesting part: bootstrap resamples hold ~63% unique rows, and some replicates converge
into the **second seductive optimum** on the likelihood surface — the one the cold-fold
analysis independently found at train 2.56 / held-out 13.05. Two unrelated procedures
(10-fold cold refits, and 360 bootstrap fits) both fall into it. So the likelihood surface
has a competitive-looking basin that fits ~135 rows better and generalises far worse, and
any resampling scheme on this dataset must be checked for it. That is a statement about the
problem, not about our code, and it belongs in the deck.

### Where the ODE itself is off — independent of the tree

In the same 0.5 < p ≤ 10 band the physics **misses low**, and this holds whether or not the
tree is in the model. But state it carefully: **mean +4.34, median +1.05, 8/13 rows
under-predicted, band RMSE 11.26 vs 5.64 overall.** That is *not* a uniform 4-point offset —
it is a few badly-missed rows dragging the mean. Anyone who asks "bias or two outliers?"
is right to, and the honest answer connects it straight to the
29%-of-rows-carry-91%-of-the-error decomposition rather than standing as a separate claim.
Same caution on "supplies +3.29 of +4.34 needed": that is a mean-on-mean ratio over a skewed
distribution, so it is a qualifying clause, never a headline.

### Answer ready for "why is there a random forest inside your physics model?"

> Our ODE misses low in the 0.5–10% yield band — median about a point, mean about four,
> because a few cliff rows are badly missed. A 13% weight on a decorrelated tree recovers
> roughly three-quarters of that mean. We could not demonstrate that this is bias correction
> at our pre-registered bar — 8 of 13 on the sign test against a bar of 11 — and at n = 13
> that test only detects a very large effect, so we claim the weaker interpretation: variance
> reduction from a decorrelated component. The cost is real: on the near-zero rows slice RMSE
> goes 0.37 to 1.96, and overall it introduces +1.08 of bias. We keep it because the net is
> +0.27 across all three held-out seeds on a plateau in w, and because we tested the
> principled alternative — bagging the physics fit — which recovered only +0.13 and
> destabilised one seed.

Deliver this deliberately; do not get discovered by it.

**Test-set coverage of the blend region.** 36 of 50 test rows fall below the cutoff (72%) vs
98 of 150 train (65%), and the band composition matches: dead 44% test vs 37% train, low
8.0% vs 8.7%, mid 20% vs 19%. So the gain was measured against a population resembling the
one we face. **Do not claim a helpful:harmful ratio on the test set** — helpfulness needs
labels, so it is not a measurable quantity there. On train it is 28:70 by error-reduced or
39:59 by pull-sign; the two definitions disagree, so name which one you mean.

**Known cost, accepted deliberately.** Noise-averaging slightly worsens the dead-regime edge:
one training row (index 97, true **0.282**, raw 0.000 → smoothed 2.549) is lifted off zero.
That row is an *edge* row, not an interior one — interior dead rows are exactly 0.000 (15/15
hot-regime training rows), so a non-zero truth means smoothing is correctly expressing cliff
uncertainty and merely overshot. Cost ≈ 5 squared-error units against ≈ 616 recovered. No
dead-regime guard was added: that would be a third tuned rule on a mechanism that is already
physically justified.

`make_submission.py` applies this by default (`--no-blend` for pure physics, `--cutoff` to
override) and averages the tree over seeds 0–2 to match the estimator w was chosen against.

Residual correction (`src/residual.py`) is **rejected**, re-verified at high effort by
`scripts/verify_residual_hybrid.py`: Ridge / shallow-ExtraTrees / small-GBM correctors over
5 seeds with the ODE refit per fold gain at most **+0.027** against a 0.3 bar. Our residual
carries little learnable structure — which is why an external audit measuring against a
weaker physics fit (train RMSE 11.3) saw a large hybrid gain and we do not.

## Identifiability — profile likelihoods

`scripts/profile_params.py --param <name>` pins a parameter across a grid, refits the rest,
and reports the admissible interval (≤10% above optimum). Runs all grid points in parallel.

Quote the **F-test** intervals below, not the 10%-RMSE band the script prints. That band is
a display heuristic and is far too generous (it gave E₂ ∈ [210, 320]); the F-test criterion
is `RMSE ≤ RMSE_min · sqrt(1 + F(1, n−p, α)/(n−p))` with n=150, p=7.

| Parameter | Best | 1σ | 95% |
|---|---|---|---|
| `E2_kJ` | 250.0 | [244.6, 257.4] | **[233.8, 269.9]** |
| `a1` | −11.79 | [−11.88, −11.71] | **[−12.12, −11.49]** |
| `n_flow` | 0.00 | [−0.01, 0.00] | **[−0.03, 0.02]** |

What each settles:
- **E₂** — an audit's E₂ ≈ 155 sits at RMSE 6.28, nowhere near the interval. Excluded, not
  merely disputed. An independent blind refit (different integrator and optimizer, no access
  to our code) produced 95% [234, 271] — agreeing with ours to **1 kJ/mol on both ends**.
- **a₁** — a₁ = 0 costs **+4.54 RMSE** and is far outside the interval. The
  thermal-concentration pathway is required, not optional.
- **n_flow** — turbulent h ∝ Re^0.8 (n ≈ 0.8) costs **+11.40**. The 95% interval excludes
  even 0.03, so the jacket coupling is flow-independent in residence-time coordinates and
  the controlling thermal resistance is not on the process side.

**Do not add an ensemble over parameter uncertainty.** Tested: eight distinct admissible
starts (E₂ spanning 210–265) refit inside a fold all converge to the *same* optimum to
within 2e-4 (`scripts/weighted_ensemble.py`). The apparent spread is an artifact of pinning
during profiling. There is no posterior to average over, which matches the likelihood
arithmetic — at n=150, RMSE 3.66 vs 3.99 is a weight ratio near 2.5e-6.

## Where the error lives

Grouping training rows out-of-fold by the spread of yields among their six nearest
neighbours in (log τ, T_in, T_jacket):

| Neighbour spread | n | OOF RMSE | % of squared error |
|---|---|---|---|
| 0–20 | 47 | 0.652 | 0.4% |
| 80–101 | 44 | 10.410 | **91.0%** |

**29% of rows carry 91% of the error.** Further modelling of the well-determined majority
cannot move the score; the result is decided by how the cliff-edge rows fall. Use this to
judge whether a proposed improvement is worth the time.

`scripts/audit_rows.py` records the row-level verdicts against an external audit (rows 0,
24, 39, 41, 3, 23), each decided on training evidence rather than model preference.

`guide.md` is a team brief, not ground truth — three of its checkable claims are wrong:
it says 18% of rows exceed yield 90 (actually 12%), that test ranges sit inside train
ranges (flow 79.57 > 79.02, length 2.03 < 2.26), and that ExtraTrees+physics scores 14.02
(reproduced here at 16.37). Re-verify before quoting it.

Two interpretation traps worth keeping straight:

- **corr(log_tau, yield) ≈ 0 does not mean residence time is unimportant.** It is the
  signature of an interior optimum — low tau leaves A unconverted, high tau destroys B, so
  both tails are low yield and the linear correlation cancels.
- **"Concentration doesn't matter because both reactions are first-order" is incomplete,
  and the obvious repair is also wrong.** CA0 cancels from the *isothermal* yield
  expression, but a richer feed releases more reaction heat and should change yield through
  the thermal path. The tempting conclusion is "the reactions must be thermally neutral" —
  but the `neutral` variant (a1 = a2 = 0) was fitted and scores 8.27 vs 3.66, so the heat
  terms are real. What actually happens: a1 = −11.79 and a2 = +11.34 have **opposite signs
  and near-equal magnitude**, so at mean CA0 the adiabatic swings are −27 K then +26 K, a
  net of −1 K. The two heat effects very nearly cancel. That is the defensible answer.

## Rejected mechanisms — do not re-litigate without new evidence

Each was fitted and measured, not argued away:

| Hypothesis | Result |
|---|---|
| Flow-dependent jacket U, `(Q/Q_ref)^n` | n ≈ −0.03, inactive; +0.015 RMSE for one extra param |
| Thermally neutral reactions | 8.27 vs 3.66 — clearly worse |
| Parallel A→C path | **Now measured** (`scripts/pitch_evidence.py`). Fitted with `ln_k3_ref, E3_kJ, a3` free: train 3.6540 vs 3.6559 — 3 extra parameters buy **+0.0019**. 10f-CV over 3 seeds is **6.7103 vs 6.3579, i.e. −0.352 worse**, and unstable (5.41 / 8.13 / 6.59). Rejected on generalization |
| Axial dispersion (tanks-in-series) | At *fixed* params worth only ~0.08 RMSE. Larger apparent gains came from refitting against the coarse cascade's discretization error — same failure mode as the step-drift bug |

The self-recovery test in `diagnose_fit.py` is the tool that made these calls trustworthy:
it regenerates targets from the fitted parameters and refits from scratch. It recovers all
7 parameters to 5 decimals (RMSE 0.00000), which proves the optimizer is sound. **Run it
before concluding "the search failed" or "the model needs another parameter" — those two
look identical without it.**

### Correction: the A→C rule-out was invalid, and so was its replacement

**The original argument.** "Yields reach 99.97%, so a parallel path is excluded" only bounds
`k3/(k1+k3) < 0.0003` **at that row's temperature of 383 K**, while training temperatures
span 351.6–548.0 K. Since k₃ carries its own activation energy it can be negligible at 383 K
and significant at 548 K. Locally true, globally invalid — a *lucky assertion*, not a
measurement. It reached the right conclusion by an argument that does not support it.

**The replacement prose was also wrong.** For a while the notebook asserted "ln k₃ drives to
its lower bound and cross-validated RMSE moves by 0.0003, every other parameter unchanged to
8 significant figures." No artifact ever backed it, and when finally measured
(`scripts/pitch_evidence.py`) **all three details were false**: ln_k₃ lands at −7.16 on one
DE seed and −11.65 on another (bound is −12, so not reliably at it), the CV delta is
**−0.352, not 0.0003**, and E₂ shifts by 1.08 kJ/mol with a₂ by 0.286 — not eight-figure
agreement.

**What is actually true, and it is a better argument.** The fitted A→C model buys **+0.0019**
train RMSE for three extra parameters (3.6540 vs 3.6559) while 10-fold CV over three seeds
gets **worse: 6.7103 vs 6.3579**, swinging 5.41 / 8.13 / 6.59 across seeds. Negligible train
gain plus degraded, unstable generalization is the signature of parameters fitting noise. The
data supports the series network.

Worth stating in the pitch: the problem statement itself calls the network *"series-parallel"*
while listing only A→B→C. We tested the parallel path rather than assuming either reading.
The A→C integrator is exact, not an approximation — A decays at `kA = k1 + k3` while B is
produced at `k1`, so the analytic step generalises; it reduces to the shipped model to
**5.2e-12** yield-points at k₃→0 and agrees with scipy BDF to **9.1e-04** with k₃ active.

## Input noise: what is measured, what is inferred

An earlier framing ran through the whole project — "the data is deterministic simulator
output, so the theoretical best RMSE is ~0, and any error is *our* modelling error." That
was wrong and made every plateau look like missing physics. But the replacement claim ("we
are at the noise floor") was also overstated. What actually holds:

**Strong evidence (measured):**
- **Noise-averaging improves held-out error.** Predicting `E[f(x+δ)]` with δ ~ N(0, σ²) on
  both temperatures moves 10-fold CV from **6.358 → 5.874** at σ = 2.5 K, better on all
  three seeds, while making the *training* fit worse (3.656 → 3.867). Blurring a correct
  model with clean inputs would hurt CV, not help it. This is the single strongest result
  (`scripts/noise_averaged.py`).
- **No residual structure.** A scan over ~104 features and pairwise products finds nothing
  (largest |r| = 0.137, expected false positives ≈ 0.2).
- **Errors-in-variables offsets are structureless.** Solving for the per-row temperature
  offset that reproduces each observation exactly (`scripts/eiv_noise.py`) gives median
  |Δ| = **1.67 K** with largest feature correlation 0.206.

**Weak or superseded evidence — do not lean on these:**
- ~~Residuals are unbiased~~. Least squares drives the residual orthogonal to ∂f/∂θ, so a
  small mean residual is a **first-order condition of the optimizer**, not evidence about
  noise. This was wrongly presented as an independent pillar.
- **Sensitivity-scaling rules out *output* noise but cannot separate input noise from
  misspecification in the temperature channel** — a small error in E₂ produces the same
  dYield/dT signature. It is consistent with the noise story, not diagnostic of it.

**What does not fit a clean noise model.** The EIV offsets have std **5.53 K** with a 99th
percentile of **22 K**, and **12 of 150 rows** cannot be reproduced by any offset within
±25 K. A σ ≈ 2 K Gaussian would have a 99th percentile near 5 K. So a minority of rows carry
something that is not temperature noise.

**Arithmetic reconciliation.** Injecting N(0, 2.24 K) produces a *train-equivalent*
prediction spread of 4.29. Compared against CV 6.358, the non-noise component is
`sqrt(6.358² − 4.29²) ≈ 4.7`; against the smoothed CV 5.874 it is ≈ 4.0. That remainder is
parameter-estimation variance (7 parameters from 135 rows, with the cliff-region parameters
weakly determined — see the cold-fold spreads) plus whatever the 12 unreachable rows are.
**Do not claim CV is irreducible.**

**Assumption, flagged as such.** Noise-averaged prediction is optimal *only if* the hidden
targets are `f(true inputs)` while we are given noisy inputs. The evidence is indirect — the
training residual is non-zero, so something differs between the published inputs and
whatever generated the targets, but that could be misspecification instead. The empirical
case does not depend on the mechanism: smoothing improves honest held-out error by 0.48, and
CV is a faithful proxy for the test set under either explanation.

## Out of scope (rubric explicitly penalizes brute force)

Neural nets (150 rows), XGBoost/LightGBM hyperparameter searches (boosting measured
*worst* here), polynomial feature explosion, log/Box-Cox target transforms (cannot
represent the 24.7% exact zeros), outlier removal (the extreme rows are real physics
regimes and carry the location of the yield cliff).

## Validation rule

Repeated K-fold across ≥5 seeds, always. A single split at n=150 moves by several RMSE
points with the seed. Blend weights are chosen by minimizing out-of-fold RMSE, never by a
gut number — the brief's suggested fixed 70/30 would more than double the error if the
physics model is much stronger than the tree.
