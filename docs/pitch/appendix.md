# Technical Appendix — Team *Claude ke Chatore*

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

| Band | n | mean pull | applied | needed | band RMSE gain |
|---|---|---|---|---|---|
| dead, p ≤ 0.5 | 56 | +5.92 | +0.77 | +0.13 | **-1.55** |
| low, 0.5 < p ≤ 10 | 13 | +25.28 | +3.29 | +4.34 | **+1.74** |
| mid, 10 < p ≤ 60 | 29 | +5.38 | +0.70 | -2.41 | +0.23 |

The entire gain is 13 rows.

**The tree is variance reduction, not bias correction.** Sign test on the band carrying the
gain: **8** of **13** aligned (62%, p = **0.581**) against a pre-registered bar of ≥ 11/13.
Per seed 7/12, 9/13, 9/14 — none clears it. Correlation between physics error and tree error
is **+0.070**; the tree scores **15.04** standalone on those rows against 6.14 for physics.

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
| File | `claude_ke_chatore.csv` |
| sha256 (first 16) | `c68e0e748e4928f2` |
| Rows | 50, in `test_dataset.csv` order |
| Column | one, header `overall_yield` |
| Range | [0.005, 97.567], mean 29.413 |
| Built by | `scripts/make_submission.py` at 2048 substeps |

`src.data.write_submission` asserts the contract and re-reads the file to re-validate.
