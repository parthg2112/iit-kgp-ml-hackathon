# Technical Appendix — Team *Claude ke Chhatore*

Verified by `python scripts/audit_pitch.py`, which reads every figure below live from
`artifacts/` and fails if any has drifted.

---

## 1. The model

```
dCA/dz = -k1*CA
dCB/dz =  k1*CA - k2*CB
dT/dz  =  a1*k1*CA + a2*k2*CB + U*(T_jacket - T)

CA(0) = CA0,  CB(0) = 0,  T(0) = T_inlet,  integrated to tau = length / flow
yield = 100 * CB(tau) / CA0
```

Arrhenius reparameterized about **T_ref = 430 K**: `k = exp(ln_k_ref) * exp(-E/R * (1/T - 1/T_ref))`.
Fitting ln A and E directly correlates them above 0.999.

**Note on units:** `tau = L/Q` has units m·min/L, not time — the true residence time is
`A_c·L/Q`. Tube diameter is constant across all rows, so `A_c` folds into the fitted
pre-exponentials and into U. Harmless for prediction; it means a different tube diameter
requires a refit rather than a rescale.

## 2. Fitted parameters

Model `series`, 7 free parameters, `n_flow` pinned to 0.

| Parameter | Value | Units | Interpretation |
|---|---|---|---|
| `ln_k1_ref` | 2.7187 | — | ln of the A→B rate constant at 430 K |
| `E1_kJ` | **43.16** | kJ/mol | activation energy, desired reaction |
| `ln_k2_ref` | 0.1594 | — | ln of the B→C rate constant at 430 K |
| `E2_kJ` | **250.07** | kJ/mol | activation energy, waste reaction |
| `a1` | **-11.79** | K·L/mol | adiabatic temperature coefficient, A→B |
| `a2` | **+11.34** | K·L/mol | adiabatic temperature coefficient, B→C |
| `U` | **3.2552** | 1/(m·min/L) | jacket heat-transfer coefficient |

Derived: **E2/E1 = 5.79** · k1 = k2 crossover at **449.9 K** · training temperatures span
351.6–548.0 K, so the crossover is interpolated.

`check_bounds.py` refit with the box widened to E1 ∈ [5, 400], E2 ∈ [20, 600] and returned
the same optimum with nothing at a constraint.

## 3. Identifiability — profile likelihoods

Each parameter pinned across a grid, all others refit. Intervals by the **F-test**
criterion `RMSE <= RMSE_min * sqrt(1 + F(1, n-p, alpha)/(n-p))` with n = 150, p = 7.

| Parameter | Best | 1 sigma | 95% |
|---|---|---|---|
| `E2_kJ` | 250.07 | [244.63, 257.35] | **[233.75, 269.92]** |
| `a1` | -11.79 | [-11.88, -11.71] | **[-12.12, -11.49]** |
| `n_flow` | 0.00 | [-0.01, 0.00] | **[-0.03, 0.02]** |

**Do not quote the 10%-RMSE band.** `profile_params.py` also prints an "admissible" range at
RMSE ≤ 1.10 × optimum — for E2 that display heuristic spans 210 to 320. It shows the basin
is broad, and the flat bottom is a genuine robustness observation, but it is **not** a
confidence interval and is far too generous. The F-test intervals above are the statistical
statement.

What each settles:

- **E2** — an external audit's E2 ≈ 155 sits at train RMSE 6.28, **+2.62** above optimum and
  nowhere near the interval. Excluded, not merely disputed.
- **a1** — a1 = 0 costs **+4.54** train RMSE. The thermal-concentration pathway is required.
- **n_flow** — turbulent h ∝ Re^0.8 (n ≈ 0.8) costs **+11.40**. The interval excludes even
  0.03, so jacket coupling is flow-independent in residence-time coordinates.

**No ensemble over parameter uncertainty.** Eight admissible starts spanning E2 = 210–265,
refit inside a fold, converge to the same optimum within 2e-4. The apparent spread is an
artifact of pinning during profiling; there is no posterior to average over.

## 4. Validation protocols — definitions and results

**These are not comparable to one another. Every figure carries its protocol.**

| Protocol | Definition |
|---|---|
| `train` | fit and scored on all 150 rows |
| `10f-CV` | repeated 10-fold, ODE parameters refit inside every fold |
| `cold` | folds refit by differential evolution from scratch, no warm start |
| `LOSO` | prediction policy chosen on two seeds, scored on the third |
| `bootstrap` | resampled 50-row draws from out-of-fold predictions |
| `split-half` | policy chosen on 75 rows, scored on the other 75 — the only protocol here that genuinely prices selection |

**LOSO does not price hyper-parameter selection, and this appendix used to claim it did.**
All three seeds re-partition the same 150 rows, so the held-out seed has already seen every
row. Measured: choosing (w, cutoff) on 75 rows and scoring on the other 75 costs **+0.371
RMSE** against the fixed (0.87, 60), and selection loses **200 of 200** replicates; LOSO
prices the same choice at +0.000 / +0.029 / +0.022. **Read 5.671 as a lower bound.** The
policy itself is sound — every search lands at w ∈ [0.85, 0.89] with cutoff 60, and the
full-data argmax (0.86, 60) scores 5.6546 against 5.6547 for the shipped value.

**`n_flow` floats during CV but is pinned in the shipped model.** `make_physics_predict_fn`
passes the full 8-parameter box and n_flow settles at −0.02…−0.03 in every fold, so the CV
and LOSO figures describe 8 free parameters for a 7-parameter shipped model. Immaterial
(95% interval [−0.03, 0.02]) but stated rather than hidden.

| Quantity | Protocol | Value |
|---|---|---|
| ExtraTrees + physics features | 10f-CV | **16.37** ± 1.38 |
| Physics ODE, raw | train @2048 substeps | **3.6559** |
| Physics ODE, raw | train @512 substeps | 3.6617 |
| Physics ODE, raw | 10f-CV | 6.358 ± 0.124 |
| Physics ODE, raw | cold | 2.97 / 13.05 / 1.93, mean **5.98** |
| Physics ODE, noise-averaged sigma=1.67 | 10f-CV | 5.935 |
| Previous policy (raw, w = 0.910, cutoff 60) | LOSO | **6.022** |
| **SHIPPED: sigma 1.67, w 0.87, cutoff 60** | **LOSO** | **5.671** |
| Score a 50-row test set could produce | bootstrap | 5th–95th pct **[2.41, 9.07]** |

Warm-started 10f-CV (6.36) and cold folds (5.98) agree, which is the honest generalization
estimate. An earlier run reporting `3.662 ± 0.000` was a bug — a zero seed-to-seed spread
meant the folds could not move away from the full-data optimum, so they were not independent
of the rows scoring them.

**Cold-fold parameter spread** tells us which reaction is well determined: E1 and ln_k1 shift
about 4% across folds, but a2 shifts 91% and ln_k2 61%. The desired reaction is tightly
determined; the waste reaction is not, because k2 is either negligible or overwhelming across
most sampled conditions. The governing ratio k2/k1 stays stable at 0.063–0.079 regardless.

## 5. The shipped prediction policy

**(sigma, weight, cutoff) = (1.67 K, 0.87, 60)**, chosen **jointly** — optimising down a
chain lands wherever the ordering puts you.

**sigma is fixed a priori** at 1.67 K, the errors-in-variables median, *not* tuned on the CV
that then judges it. The in-sample argmax (sigma = 2.0) scores better, but quoting it would
report a gain measured at its own optimum.

**Regime-aware blend:** mix below a predicted yield of 60, pure physics above. A single
global weight hid a defect — the tree helps in a narrow band and hurts at both ends.

### 5a. Reconciled accounting — where the gain actually comes from

**RMSE is not additive; squared error is.** Per-band RMSE deltas cannot be summed, and
reading them as contributions is what produced an earlier claim in this package that "the
entire gain comes from 13 rows." **That claim was wrong.** The table below does the
accounting in SSE, where the band contributions sum to the total by construction.

Basis: pooled over 3 seeds × 150 rows = 450 predictions, which is the basis that closes
exactly. (The headline `5.9352 → 5.6541` is per-seed RMSE then averaged; pooling gives
`5.9355 → 5.6547`. Averaging RMSEs is not the same as pooling SSE, so the two differ in the
third decimal. Both are correct; they answer slightly different questions.)

| Band | n | RMSE pre | RMSE post | ΔRMSE | SSE pre | SSE post | **ΔSSE** | **% of gain** |
|---|---|---|---|---|---|---|---|---|
| dead, p ≤ 0.5 | 168 | 0.366 | 1.960 | **-1.594** | 22.6 | 645.4 | **-622.9** | **-42.5%** |
| low, 0.5 < p ≤ 10 | 39 | 11.210 | 9.580 | +1.630 | 4900.5 | 3579.2 | **+1321.3** | **+90.2%** |
| mid, 10 < p ≤ 60 | 86 | 9.441 | 8.957 | +0.484 | 7665.0 | 6899.1 | **+765.9** | **+52.3%** |
| above cutoff, p > 60 | 157 | 4.561 | 4.561 | 0.000 | 3265.4 | 3265.4 | 0.0 | 0.0% |
| **TOTAL** | **450** | **5.935** | **5.655** | **+0.281** | **15853.4** | **14389.1** | **+1464.3** | **100%** |

Closure residual: **4.6e-13**. Reproduce with `python scripts/blend_accounting.py`.

**What this corrects.** The mid band is a *major* positive contributor — **+52.3%** of the
gain — despite a small RMSE delta, because it holds 86 rows with a large baseline error
(RMSE 9.44). The low band supplies 90.2%, and the dead band gives back 42.5%. The gain is
**not** confined to 13 rows.

**On the dead band, stating the level not just the delta.** Slice RMSE goes from **0.366 to
1.960** — the blend makes those rows worse by **1.594 RMSE**, costing **622.9 SSE**. Earlier
wording ("loses -1.55 RMSE") was ambiguous between a delta and a level; it is a delta.

### 5b. Why the mean misleads on the gain-carrying band

The 0.5 < p ≤ 10 band has mean need **+3.99** but median **+0.70**. It is heavily
right-skewed, and squared error is tail-dominated:

| |need| quantile | p50 | p75 | p90 | max |
|---|---|---|---|---|---|
| yield points | 1.46 | 3.70 | 21.88 | 38.17 |

**The top 3 of 39 rows supply 1016 of the band's 1321 SSE improvement (77%).** The five
largest are rows the physics puts near zero while truth is 28–39:

| y | physics | tree | blended | err pre | err post | ΔSSE |
|---|---|---|---|---|---|---|
| 28.73 | 5.160 | 87.86 | 15.911 | +23.57 | +12.82 | +391.3 |
| 38.84 | 0.671 | 38.27 | 5.559 | +38.17 | +33.28 | +349.3 |
| 38.84 | 4.497 | 37.45 | 8.781 | +34.34 | +30.06 | +275.9 |
| 38.84 | 5.861 | 40.25 | 10.332 | +32.98 | +28.51 | +274.9 |
| 28.73 | 7.275 | 66.66 | 14.995 | +21.46 | +13.74 | +271.7 |

This is the skew that reconciles the arithmetic: a mean need of ~+4 alongside an SSE gain
that implies much larger per-row errors. Both are true because a handful of cliff rows carry
errors of 20–38 yield-points.

**The tree: what we can and cannot demonstrate.** We could not demonstrate bias correction
at our pre-registered bar — sign test on the gain-carrying band gives **8** of **13** aligned
(62%, p = **0.581**) against a bar of ≥ 11/13, and per seed 7/12, 9/13, 9/14, none clearing
it.

**This is a failure to demonstrate, not a demonstration of absence.** At n = 13 the sign test
only has power against a very large effect, and the *magnitude* evidence points the other
way: the tree supplies **+3.29** where **+4.34** is needed — same sign, about 76% of the
required correction. We cannot separate that from variance reduction at this sample size, so
we claim the weaker interpretation and say so explicitly.

What is measured either way: correlation between physics error and tree error is **+0.070**,
and the tree scores **15.04** standalone on those rows against 6.14 for physics — a weak,
decorrelated component, which is exactly what improves an ensemble by averaging.

*Beware `corr(tree - physics, y - physics)`* — both share the `- physics` term, so it reads
**+0.389** and is spuriously high. We report it here only to say we did not use it. The sign
test is the clean statistic, and it must be run on a stratum where both `need` and `pull`
vary in sign (in the dead band the tree's pull is positive on 56 of 56 rows, so the test
there is near-tautological; its RMSE cost of -1.55 is the meaningful statement).

**Cost of the blend:** introduced bias **+1.083** on blended rows (+0.066 → +1.148). Accepted
because the net is +0.27 across all held-out seeds and the weight sits on a plateau (gain
0.281 / 0.279 / 0.271 at w = 0.87 / 0.85 / 0.89).

**Test-set coverage:** 36 of 50 test rows fall in the blend region (72%) vs 98 of 150 train
(65%), and the band composition matches — dead 44% vs 37%, low 8.0% vs 8.7%, mid 20% vs 19%.
Helpfulness on the test rows is *unknowable without labels* and we do not quote it.

## 6. Rejected alternatives, with protocol

| Hypothesis | Result | Protocol |
|---|---|---|
| Thermally neutral (a1 = a2 = 0) | **8.27** vs 3.66 | train |
| a1 = 0 alone | +4.54 | train |
| Parallel A→C (ln_k3, E3, a3 free) | train +0.0019; CV **-0.352**, seeds 5.41 / 8.13 / 6.59 | train / 10f-CV, 3 seeds |
| Flow-dependent U, n = 0.8 | +11.40 | train |
| Free reaction orders n1, n2 | +0.072, one seed -0.189 | 10f-CV, 5 seeds |
| Axial dispersion (tanks-in-series) | ~0.08 at fixed parameters | train |
| Residual ML corrector | +0.027 vs a 0.3 bar, 3 correctors | 10f-CV, 5 seeds |
| Bagged physics as a tree replacement | recovers +0.13 of +0.28; one seed worse | 10f-CV, 3 seeds |
| Ensemble over parameter uncertainty | degenerate, spread 2e-4 | — |
| E2 = 155 (external audit) | +2.62 | train |

**On the A→C test specifically:** the integrator is exact, not an approximation. A decays at
`kA = k1 + k3` while B is produced at `k1`, so the analytic step generalises. It reduces to
the shipped model to **5.2e-12** yield-points at k3 → 0, and agrees with SciPy BDF to
**9.1e-04** with k3 active.

## 7. Input noise — what is measured, what is inferred

**Measured:**

- Residual scales with input sensitivity: least-sensitive quintile RMSE 0.200, most-sensitive
  7.393. Rules out *output* noise.
- No residual structure across 104 features and pairwise products (largest |r| = 0.137,
  expected false positives ≈ 0.2).
- Errors-in-variables: a median temperature offset of **1.67 K** reproduces **138 of 150**
  rows; offsets are structureless (largest feature correlation 0.206).
- Noise-averaging degrades train (3.6559 → 3.7643) while improving 10f-CV
  (6.3579 → 5.9279), better on all three seeds.

**Not claimed:**

- **12 of 150 rows** cannot be reproduced by any offset within ±25 K. Unexplained. The EIV
  offsets have std 5.53 K with a 99th percentile near 22 K, where a clean 2 K Gaussian would
  give about 5 K — so a minority of rows carry something that is not temperature noise.
- The residual is **not** fully accounted for. Against 10f-CV 6.36 the non-noise component is
  roughly 4.7: parameter-estimation variance plus those 12 rows.
- Residual unbiasedness is **not** evidence — least squares forces orthogonality to
  df/dtheta, making it a first-order condition of the optimizer.
- Sensitivity scaling **cannot separate input noise from misspecification**: a small error in
  E2 produces the same dY/dT signature.

**Stated assumption:** noise-averaged prediction is optimal only if the hidden targets are
`f(true inputs)` while we are given noisy inputs. The empirical case does not depend on the
mechanism — smoothing improves honest held-out error by 0.48 either way.

**Known cost, accepted:** training row 97 (true **0.282**) is lifted from raw 0.000 to
smoothed 2.549. Its truth is non-zero, marking it an *edge* row rather than an interior one —
interior dead rows are exactly 0.000 in 15 of 15 hot-regime cases — so the lift is smoothing
correctly expressing cliff uncertainty and merely overshooting. Cost ≈ 5 squared-error units
against ≈ 616 recovered. No dead-regime guard was added; that would be a third tuned rule on
a mechanism already physically justified.

## 8. Numerics

- **Operator splitting**, rates frozen per substep: mass balance advanced analytically (exact
  solution of the linear series reaction), energy balance with the exact linear-ODE solution.
  Both unconditionally stable — no step size produces negative concentrations or oscillation.
  A midpoint predictor–corrector makes the scheme second order.
- **Fixed-step RK4 was tried and abandoned** — it blew up on thermal-runaway rows.
- **Verified against SciPy BDF** (`check_integrator.py`), agreeing to 0.017 yield-points at
  2048 substeps.
- **Timing, measured:** 24.8 ms per 150-row evaluation at 512 substeps, 102 ms at 2048, vs
  5573 ms for the BDF reference — a **225x** speed-up at 512.
- **Convergence guard:** `residuals()` can reject parameter sets whose integration has not
  converged, by re-integrating at half resolution. It defaults **off** and is passed only at
  differential-evolution call sites — it is a cliff in an otherwise smooth objective, and a
  finite-difference Jacobian straddling it is meaningless.
- **Self-recovery test:** regenerating targets from the fitted parameters and refitting from
  scratch recovers all 7 parameters to 5 decimals (RMSE 8e-09). This is what makes "the model
  needs another parameter" distinguishable from "the search failed".

## 9. Submission

| | |
|---|---|
| File | `claude ke chhatore.csv` |
| sha256 (first 16) | `c68e0e748e4928f2` |
| Rows | 50, in `test_dataset.csv` order |
| Column | one, header `overall_yield` |
| Range | [0.005, 97.567], mean 29.413 |
| Built by | `scripts/make_submission.py` at 2048 substeps |

`src.data.write_submission` asserts the contract and re-reads the file to re-validate.
