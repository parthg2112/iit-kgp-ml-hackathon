# Reactor Yield Surrogate — Team *Claude ke Chhatore*

IIT-KGP ML Hackathon: predict `overall_yield` of product B from five reactor operating
conditions. 150 training rows, 50 test rows, scored on RMSE. **One submission, no
leaderboard feedback.**

**Approach in one line:** we did not fit a regressor to the target — we recovered the
reactor's governing differential equations and fitted their seven physical parameters. A
13% weight on an ExtraTrees model rides along for variance reduction; the mechanism is the
ODE.

### What ships

`claude ke chhatore.csv`, built by `scripts/make_submission.py` from the policy triple
**(σ, w, cutoff) = (1.67 K, 0.87, 60)** stored in `artifacts/blend.json`: physics predictions
noise-averaged over σ = 1.67 K, blended 0.87/0.13 with the tree below a predicted yield of
60, pure physics above.

**Every number below carries its protocol — they are not comparable.**
`train` = fit and scored on all 150 rows. `10f-CV` = repeated 10-fold, parameters refit per
fold. `LOSO` = policy chosen on two seeds and scored on the third — the headline figure,
but **read it as a lower bound**: all three seeds re-partition the same 150 rows, so LOSO
prices fold-partition noise, not hyper-parameter selection. A row-level split-half puts the
selection cost at **+0.371**. See CLAUDE.md.

| Model | Protocol | RMSE |
|---|---|---|
| Best tree ensemble (ExtraTrees + physics features) | 10f-CV | 16.37 ± 1.38 |
| Physics ODE, raw | 10f-CV | 6.36 ± 0.12 |
| Physics ODE, noise-averaged σ=1.67 | 10f-CV | 5.94 |
| Previous policy (raw, w=0.910) | LOSO | 6.022 |
| **SHIPPED: σ=1.67, w=0.87, cutoff=60** | **LOSO** | **5.671** |
| Physics ODE, training fit (2048 substeps) | train | 3.6559 |

A 50-row test set carries large sampling noise on top of any of these: bootstrapping the
out-of-fold predictions gives a 5th–95th percentile of **[2.41, 9.07]**.

---

## Where things are

```
claude ke chhatore.csv     <- THE SUBMISSION. 50 rows, one column. Do not hand-edit.
notebook/final.ipynb      <- the finalist deliverable; self-contained, runs top to bottom
README.md                 <- you are here
CLAUDE.md                 <- engineering notes, gotchas, and every rejected hypothesis

data/                     train_dataset.csv, test_dataset.csv (unmodified)
docs/                     problem statement PDF, original team brief (guide.md)
src/                      the model — see below
scripts/                  everything runnable; each is one experiment
artifacts/                fitted parameters, CV results, profiles (JSON/npy)
figures/                  diagnostic plots
reference/                an external audit's competing predictions, kept for comparison
```

### `src/` — the model

| File | What it holds |
|---|---|
| `physics.py` | The reactor ODE, the custom integrator, parameter bounds and model variants. **The core.** |
| `data.py` | Loading, physics feature engineering, and the submission writer (which asserts the contract) |
| `baseline.py` | ExtraTrees safety net — a comparison point, not the model |
| `residual.py` | Physics + ML residual correction. Measured and **rejected** (gain ≤ 0.027) |
| `evaluate.py` | Repeated K-fold harness, blend-weight search, regime-aware blending |

### `scripts/` — each answers one question

```
check_integrator.py       is our solver correct?         (gate: agrees with SciPy BDF)
fit_physics.py            fit the 7 parameters
compare_models.py         which model variant does the data support?
profile_params.py         how well is a parameter determined?   --param E2_kJ | a1 | n_flow
run_baseline_cv.py        how good is a plain ML baseline?
run_physics_cv.py         honest cross-validation + blend search
check_cold_folds.py       generalization without warm-start optimism
diagnose_fit.py           is a bad fit the search's fault or the model's?
audit_rows.py             adjudicate disputed test rows against training data
make_submission.py        build the CSV (asserts the contract, re-reads to verify)
build_notebook.py         regenerate notebook/final.ipynb
```

---

## Running anything

The default `python` (3.13) has **no scientific stack**. Use 3.11:

```bash
PY="C:/Users/USER/AppData/Local/Programs/Python/Python311/python.exe"
$PY scripts/check_integrator.py     # start here — verifies the core is sound
$PY scripts/make_submission.py      # rebuild the CSV
```

Run scripts from the repo root. Long fits: use `-u` and redirect to a log, or output buffers
until the end.

---

## The physics, briefly

Two first-order reactions in series inside a tube:

```
A --k1--> B --k2--> C          B is the product AND the feedstock for the waste reaction
```

Both rates are Arrhenius. We recovered **E₁ = 43 kJ/mol** and **E₂ = 250 kJ/mol** — the
waste reaction is far more temperature-sensitive, so heating destroys selectivity. The two
rates cross at **~450 K**, which is the operating limit: below it the reactor makes B faster
than it destroys it, above it the reverse.

Yield is governed by a ridge — too cold or too fast and A never converts; too hot or too slow
and B is destroyed. A quarter of the training rows are *exactly* zero (dead reactor).

## Three things that will trip you up

1. **Never pass a lambda to `differential_evolution`** — it can't be pickled, so `workers=-1`
   silently fails and the fit uses 1 of 12 cores.
2. **The convergence guard in `residuals()` defaults OFF and must stay off during
   `least_squares` polish** — it's a cliff in the objective and wrecks the Jacobian.
3. **`guide.md` (in `docs/`) is a team brief, not ground truth.** Several of its numbers do
   not reproduce. Re-measure before quoting it.

Full detail on all three, plus every hypothesis we tested and rejected, is in `CLAUDE.md`.
