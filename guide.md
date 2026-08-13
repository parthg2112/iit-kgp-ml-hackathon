# ML Hackathon — Team Brief
### Predictive Modeling Optimization Challenge (Reactor Yield Surrogate)

**Read time: ~10 minutes. Read this before writing any code.**

---

## 1. What we are actually being asked to do

A chemical plant has a rigorous physics simulator that predicts reactor performance. It works, but it is slow (differential equations, minutes per run). Plant engineers cannot use it for real-time decisions.

Our job is to build a **surrogate model** — a fast approximation of that simulator, learned from 150 historical data points. Give it 5 operating conditions, it instantly returns the predicted yield of the desired product.

| Item | Detail |
|---|---|
| Train data | 150 rows, 5 inputs + 1 target |
| Test data | 50 rows, inputs only |
| Target | `overall_yield` — % of Product B at reactor exit |
| Metric | RMSE against a hidden "true physics" answer key |
| Submissions | **One only.** No leaderboard feedback. |
| Deliverable | `[TeamName].csv`, exactly 50 rows, exactly one column `overall_yield`, floats to 3+ decimals, same row order as test file |

Scoring is two-phase: RMSE gets us into the top 6, then a live pitch decides the winner. The problem statement says plainly that brute-forcing algorithms without understanding the chemistry will not win. **Both halves matter.**

---

## 2. The chemistry, in plain terms

Two reactions happen in series inside a tube:

```
A  --k1-->  B  --k2-->  C
           (want)      (waste)
```

We want to maximise B. But B is an intermediate — it is being made and destroyed at the same time. This is the classic "series reaction" trade-off.

Both rate constants follow Arrhenius:

```
k = A · exp( -E / (R·T) )
```

So both speed up when we heat the reactor. The catch: **k2 almost certainly has a higher activation energy than k1**. That means heating helps the bad reaction more than the good one. Push the temperature and you convert all your A into B, then immediately burn all your B into C.

There are two knobs that control this:

- **Temperature (T)** — sets how fast the reactions go. Controlled by `inlet_temperature_K` and `jacket_temperature_K`.
- **Residence time (tau)** — how long the fluid stays in the tube. This is `length_m / flow_rate_L_min`. Long reactor or slow flow means more time to react.

The yield surface therefore has a **ridge**. Too cold or too fast, A never converts, yield is low. Too hot or too slow, B gets destroyed, yield is zero. Optimum sits on a narrow band between them.

The textbook expression for an isothermal plug-flow reactor is:

```
Yield_B = k1/(k2 - k1) · ( exp(-k1·tau) - exp(-k2·tau) )
```

Note what is **missing** from that formula: the inlet concentration. It cancels out algebraically when both reactions are first-order. Remember this — it matters in Section 4.

Our reactor is described as *non-isothermal*, so temperature is not constant along the tube. It starts at `inlet_temperature_K` and is dragged toward `jacket_temperature_K` by heat exchange through the wall, plus whatever heat the reactions themselves release. That is why we get inlet and jacket as separate features.

---

## 3. Feature glossary

| Column | Meaning | Physical role |
|---|---|---|
| `flow_rate_L_min` | Volumetric flow (L/min) | Denominator of residence time. Faster = less reaction time. Also affects how much heat the jacket can transfer per unit of fluid. |
| `concentration_mol_L` | Inlet concentration of A | Sets absolute output. Also sets how much reaction heat is released. |
| `inlet_temperature_K` | Feed temperature | Starting point of the thermal profile |
| `length_m` | Reactor length | Numerator of residence time. Longer = more reaction time. |
| `jacket_temperature_K` | Heating jacket temperature | The temperature the fluid is being pulled toward |
| `overall_yield` | **Target**, 0–100 | % of A that ended up as B |

---

## 4. What the data actually told us

These are measured facts from the 150 training rows, not assumptions.

### 4.1 Data is clean but tiny
Zero missing values, zero duplicate rows, and test feature ranges sit inside train ranges (no extrapolation risk). But 150 rows across 5 dimensions is *very* small. **Overfitting is our single biggest threat**, not model capacity.

### 4.2 The target is not a normal continuous variable

| Bucket | Share of rows |
|---|---|
| Exactly 0.000 | **24.7%** |
| Below 1.0 | 38% |
| Above 90 | 18% |

It is **zero-inflated and bimodal**. A quarter of the dataset is dead reactor — B completely destroyed. Any model that assumes a smooth bell-shaped target will fight the data.

### 4.3 Temperature dominates, and the sign is negative

| Feature | Pearson vs yield | Spearman vs yield |
|---|---|---|
| `jacket_temperature_K` | **−0.50** | **−0.60** |
| `inlet_temperature_K` | −0.41 | −0.38 |
| `length_m` | +0.08 | +0.01 |
| `flow_rate_L_min` | +0.04 | +0.11 |
| `concentration_mol_L` | **+0.009** | +0.04 |

Hotter reactor, lower yield. This confirms E2 > E1 — the side reaction is winning at high temperature. Exactly what Section 2 predicted.

### 4.4 The cliff is real
Two rows that show the whole system:

| Mean temp | tau | Yield |
|---|---|---|
| 365 K | 4.49 | **99.68** |
| 496 K | 1.01 | **0.00** |

Long time at low temperature = near-perfect yield. Short time at high temperature = total wipeout. The function has a sharp transition, which is why smooth global models struggle.

### 4.5 The finding that wins us Phase 2

**Concentration has essentially no effect on yield** (correlation +0.009; tree-based feature importance 0.04, the lowest of all five).

This is not noise. It is the algebraic signature of **both reactions being first-order** — CA0 cancels out of the yield expression, as shown in Section 2.

If a judge asks "why did you not use concentration?", the answer is not "the model said so." The answer is: *first-order kinetics in both steps means yield is a function of Damköhler numbers only, and inlet concentration cancels. We verified it empirically and it held.* That is the "engineering intuition" the rubric is explicitly grading.

We also confirmed the reduced picture directly: a model using only `log(tau)`, `inlet_temperature_K` and `jacket_temperature_K` performs as well as one using all five features. **The system is effectively 3-dimensional, not 5.**

---

## 5. What we tried, what failed, what worked

All numbers below are **10-fold cross-validated RMSE on the training set** (lower is better; the target ranges 0–100).

| Approach | CV RMSE | Verdict |
|---|---|---|
| GradientBoosting, raw 5 features | 23.41 | ❌ Failing |
| RandomForest, raw 5 features | 20.47 | ❌ Failing |
| ExtraTrees, raw 5 features | 19.21 | Weak baseline |
| GradientBoosting + physics features | 18.58 | ❌ Still failing |
| RandomForest + physics features | 18.69 | ❌ Still failing |
| ExtraTrees + physics features | **14.02** | ✅ Best generic model |
| Two-stage zero-classifier × regressor | 15.22 | ✅ Idea worth keeping |

### Why the failures fail

**Boosting underperforms here.** GradientBoosting was the *worst* model tested. On 150 rows it memorises the training set before it learns the shape of the yield surface. Every gradient-boosting variant we throw at this will have the same problem — this includes XGBoost and LightGBM, so do not spend the weekend tuning them.

**Raw features hide the physics.** Handing a model `length_m` and `flow_rate_L_min` separately forces it to discover the ratio `L/Q` by itself, from 150 examples, using axis-aligned splits. It cannot. Give it the ratio directly and error drops 27%.

**Trees cannot represent a cliff smoothly.** A step function approximated by piecewise constants will always leave residual error at the transition band. This is why even our best generic model sits at RMSE 14 — a real physics model should get far below that, because the underlying data is deterministic simulator output with little or no noise.

**Key mental model:** this data has almost no randomness in it. It is a deterministic function evaluated at 150 points. That means the theoretical best RMSE is near zero, and any error we have is *our* modelling error, not irreducible noise. That is a very different game from typical Kaggle problems.

### Why ExtraTrees is the best of the generic bunch
Extra randomisation in split selection acts as a strong regulariser, which is exactly what an n=150 problem needs. It beat both RF and GBM by a clear margin.

### The features that produced the 27% gain

| Feature | Formula | Why |
|---|---|---|
| `tau` | `length_m / flow_rate_L_min` | Residence time. The most important derived quantity in the whole problem. |
| `log_tau` | `log(tau)` | Yield responds to orders of magnitude of tau, not linear tau |
| `T_avg` | `(inlet + jacket) / 2` | Proxy for mean reaction temperature |
| `delta_T` | `jacket − inlet` | Thermal driving force across the wall |
| `ln_Da_E` | `log(tau) − E/(R·T_avg)` for E = 60k, 100k, 160k J/mol | Log-Damköhler number. This is the actual dimensionless group that governs the outcome. Multiple E values let the model pick the closest match to the true activation energies. |

Anyone adding features should add them in this spirit — dimensionless groups from reaction engineering, not `feature_1 * feature_2` combinatorics.

---

## 6. Phased plan

### Phase 1 — Lock the safety net (do this first, today)
Rebuild the ExtraTrees + physics-features pipeline, predictions clipped to `[0, 100]`. Validate with **repeated 10-fold CV across multiple random seeds**, not a single train/test split — with 150 rows, a single split will lie to us. Save the predictions. This is our floor: worst case, we submit this and still score respectably.

**Owner: needs 1 person, ~2 hours.**

### Phase 2 — Fit the actual physics (highest ceiling, and the only path that wins the pitch)

Fit the real non-isothermal plug-flow reactor ODE system to our 150 rows by nonlinear least squares. The system being integrated along the reactor:

```
dCA/dt = -k1·CA
dCB/dt =  k1·CA - k2·CB
dT/dt  =  a1·k1·CA + a2·k2·CB + U·(T_jacket - T)
```

Seven parameters to recover: `ln k1`, `E1`, `ln k2`, `E2`, two heat-of-reaction terms (`a1`, `a2`), and a jacket heat-transfer coefficient `U`. Fit with `scipy.optimize.least_squares` using multi-start (the surface has local minima) and `solve_ivp` with a stiff solver.

Why this is the winning move:
- It can drive RMSE far below 14, because it is the *correct functional form*
- It handles the exact-zero rows naturally — they fall out of the equations, no special-casing
- In the pitch we report **recovered activation energies in kJ/mol**, not feature importances. We can state "E2 > E1, therefore heating destroys selectivity" and show the numbers proving it.
- It scales to real-time plant use, which the rubric explicitly asks about

*Status: this fit is currently running. It is a stiff system and slow to converge — expect it to need patience and good initial guesses. Sensible starting points: E1 around 60–90 kJ/mol, E2 around 120–200 kJ/mol.*

**Owner: strongest numerical person on the team. This is the critical path.**

### Phase 3 — Hybrid: physics + residual correction
Take the fitted physics model's predictions, compute residuals on the training set, and fit a small regularised ML model on those residuals using the physics features. Final prediction = physics + correction.

This captures whatever the simplified ODE misses (axial dispersion, temperature-dependent properties, geometry effects) without letting ML invent behaviour in regions where physics already knows the answer.

### Phase 4 — Zero-regime handling
If exact zeros are still hurting after Phase 3, add the two-stage structure: a classifier predicts P(yield ≈ 0), a regressor trained only on non-zero rows predicts magnitude, and the final output is `(1 − P_zero) × prediction`. This treats "B fully consumed" as a distinct physical regime rather than the tail of a continuum.

### Phase 5 — Blend and submit
Blend Phase 3 output with the Phase 1 safety net at roughly **70/30**. Rationale: if the ODE fit converged to a bad local minimum in a way our CV did not catch, the ExtraTrees component limits the damage. We get exactly one submission, so a small insurance premium is worth paying.

Final checklist before upload:
- [ ] Exactly 50 rows, in the **same order** as `test_dataset.csv`
- [ ] Exactly **one** column, header spelled `overall_yield` — no index column, no ID column
- [ ] All values floats, 3+ decimal places
- [ ] All values within `[0, 100]`
- [ ] File named `[TeamName].csv`
- [ ] Notebook cleaned and documented — finalists must submit the `.ipynb`

---

## 7. Explicitly out of scope

Do not spend time on these. They will not move the score and they cost us hours we need for Phase 2.

- **Neural networks / deep learning** — 150 rows
- **XGBoost / LightGBM hyperparameter searches** — boosting already tested worst here, and the rubric penalises algorithm brute-force
- **Polynomial feature explosion / automated feature generation** — adds noise dimensions to an already data-starved problem
- **Log or Box-Cox target transforms** — cannot represent the exact zeros, which are a quarter of our data
- **Outlier removal** — the extreme rows are real physics regimes, not errors. Deleting them deletes the information about where the cliff is.

---

## 8. What each of us should be able to answer in the pitch

Practice these. Phase 2 judging is on process insight, not code.

1. **Why does yield fall as temperature rises?** Because E2 > E1 — the B→C side reaction accelerates faster with temperature than A→B, so heating destroys selectivity.
2. **Why does inlet concentration not matter?** Both reactions are first-order, so CA0 cancels out of the yield expression. Verified empirically at correlation +0.009.
3. **What single quantity best explains the data?** The Damköhler number, `k1·tau`, combining residence time with the Arrhenius rate.
4. **What is the trade-off?** More residence time converts more A, but also gives B more time to degrade into C. There is an interior optimum.
5. **How did we avoid overfitting on 150 rows?** Physics-constrained functional form with 7 parameters instead of a high-variance free-form learner; repeated k-fold CV; heavy regularisation on the residual model.
6. **How would this scale to real-time plant operations?** The fitted ODE evaluates in milliseconds versus minutes for the full CFD/BVP simulation, and the parameters are physically interpretable so engineers can sanity-check them against known kinetics.

---

*Questions on any section — raise them before Phase 2 starts, not during.*