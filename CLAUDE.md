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
| ExtraTrees + physics features, repeated 10-fold | 16.37 ± 1.38 |
| Physics ODE, train (all 150 rows, 2048 substeps) | **3.6559** |
| Physics ODE, repeated 10-fold CV | **6.358 ± 0.124** |
| Physics ODE, cold-start held-out folds | 2.97 / 13.05 / 1.93 (mean 5.98) |
| **Shipped blend, 0.91 physics + 0.09 tree** | **5.711 OOF** |

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

Final model: `series` (7 free parameters, `n_flow` pinned to 0), train RMSE 3.6617.
Recovered: E1 = 43.2, E2 = 250.1 kJ/mol, a1 = −11.79, a2 = +11.34 K·L/mol, U = 3.26.
`check_bounds.py` refit with the box widened to E1∈[5,400], E2∈[20,600] and landed on the
same values with nothing at a constraint — the activation energies are real, not artifacts.

Blend search returns **w = 0.910** on physics, validated leave-one-seed-out (+0.241 vs pure
physics, positive on all three seeds; w stable at 0.878–0.919).

**But a single global weight hid a defect.** Broken down by prediction stratum the tree
*helps* mid-range and *hurts* at both ends — worst on the near-zero rows the physics gets
almost exactly right (0.319 → 1.319 RMSE over 57 rows), and again at the top where a tree
cannot extrapolate. The shipped rule is therefore **regime-aware**: blend below a predicted
yield of 60, pure physics above (`BLEND_CUTOFF` and `apply_blend` in `src/evaluate.py`).
Leave-one-seed-out: 6.022 vs 6.105 flat, better on all three seeds, and it restores the top
of the range (97.7 vs 96.6).

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

| Parameter | Admissible | Settles |
|---|---|---|
| `E2_kJ` | **[210, 320]** | An audit's E₂ ≈ 155 costs +2.62 RMSE — excluded, not merely disputed |
| `a1` | **−11.79 only** | a₁ = 0 costs **+4.54**. The thermal-concentration pathway is required, not optional |
| `n_flow` | **[−0.2, 0.0]** | Turbulent h ∝ Re^0.8 (n ≈ 0.8) costs **+11.40** — the jacket coupling really is flow-independent in residence-time coordinates |

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
| Parallel A→C path | Ruled out on data: yields reach 99.97%, a parallel path caps yield below 100% |
| Axial dispersion (tanks-in-series) | At *fixed* params worth only ~0.08 RMSE. Larger apparent gains came from refitting against the coarse cascade's discretization error — same failure mode as the step-drift bug |

The self-recovery test in `diagnose_fit.py` is the tool that made these calls trustworthy:
it regenerates targets from the fitted parameters and refits from scratch. It recovers all
7 parameters to 5 decimals (RMSE 0.00000), which proves the optimizer is sound and that
residual error is genuine model-form error. **Run it before concluding "the search failed"
or "the model needs another parameter" — those two look identical without it.**

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
