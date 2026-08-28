# Offline Round — 5 Slides, 5 Minutes
### Team *Claude ke Chhatore* · Rank 13 / 400

> **This is the presented deck.** `../support/deck.md` / `deck.html` (12 slides) is **backup depth for
> Q&A only and is not presented** — the guidelines cap the presentation at 5 slides, and
> additional slides "may not be considered during the judging process."
>
> Every number here is checked against `artifacts/` by `scripts/audit_pitch.py`.
>
> **Timings, the mark scheme and the rank below are presenter planning, not slide content.**
> `slides.html` prints none of them; it carries only a page number and, on slide 1, the team
> line. Read this file as the script, and the HTML as what the room sees.

**Mark scheme this deck is built against (200 total):** Chemical Engineering understanding
**60** · ML methodology **45** · Leaderboard **40** · Validation & failure analysis **30** ·
Presentation & Q&A **25**. Chemistry outweighs ML, and the leaderboard mark is already
banked — so slides 2 and 5 carry the most weight, not slide 3.

---

## Slide 1 — The problem (0:00–1:00)

### A reactor where the product destroys itself

```
A --k1--> B --k2--> C
          ^          ^
       product     waste
```

**B is the product *and* the feedstock for the waste reaction.** Every second B spends in the
reactor, some of it becomes C. Run too short and A never converts; run too long and B is
destroyed.

**What we predict:** `overall_yield` of B, from five operating knobs — feed rate,
concentration, inlet temperature, reactor length, jacket temperature.

**Why it matters industrially.** The rigorous simulation is, in the problem statement's own
words, *"computationally expensive and too slow for real-time plant optimization."* A plant
engineer changing a setpoint cannot wait for a boundary-value solve.

| | 150 operating conditions |
|---|---|
| Our surrogate | **24.8 ms** |
| SciPy BDF solve of the same system | 5573 ms |
| **Speed-up** | **225×**, agreeing to 0.017 yield-points |

> **Notes (60 s).** Lead with the chemistry, not the ML — the guidelines say judges should
> understand the chemical problem *before* the model. The one sentence that must land: *the
> product is also the reactant for the side reaction*, which is what makes this an optimum
> rather than a monotone response. Don't quote a runtime for the organisers' simulation — it
> was never published; our 225× is against a solver we timed ourselves.

---

## Slide 2 — The chemistry that decides everything (1:00–2:00)

### Two activation energies, and their ratio

| | Activation energy |
|---|---|
| Desired A → B | **E₁ = 43.16 kJ/mol** |
| Waste B → C | **E₂ = 250.07 kJ/mol** |

### **E₂/E₁ = 5.79** — heat accelerates destruction ~6× more steeply than formation

The rate constants cross at **449.9 K**, inside the observed envelope of 351.6–548.0 K.

**But 449.9 K is not a temperature ceiling — and that distinction is the point.** Yield
depends on k·τ, not k. Run hot with a short residence time and B leaves before it degrades:
**49 training rows sit above the crossover, and one reaches 75% yield.** The operating rule is
joint in (T, τ): *above 449.9 K, every additional unit of residence time costs selectivity.*

**Why feed concentration does almost nothing** — `corr(concentration, yield) = +0.009`:

| | |
|---|---|
| a₁ (A → B) | **−11.79** K·L/mol → −27 K |
| a₂ (B → C) | **+11.34** K·L/mol → +26 K |
| Net | **−1 K** |

Opposite signs, near-equal magnitude — the two heat effects almost exactly cancel. It is
*not* that the reactions are thermally neutral: forcing a₁ = a₂ = 0 costs train RMSE
**8.27 vs 3.66**. The heat terms are large; they oppose.

**And why residence time looks irrelevant and isn't** — `corr(log τ, yield) = +0.061`,
because an interior optimum makes both tails low-yield and the linear correlation cancels.

> **Notes (60 s).** Highest-weight slide: 60 of 200 marks. Three claims, each one a number:
> the ratio, the joint (T, τ) rule, the a₁/a₂ cancellation. If a reaction engineer challenges
> the crossover as an operating limit, agree immediately and give the 75%-yield row — we have
> tested that ourselves. The concentration argument is the one most likely to be probed: the
> tempting answer ("first-order, so CA₀ cancels") is incomplete, and the *next* tempting
> answer ("thermally neutral") is measurably wrong.

---

## Slide 3 — We recovered the reactor, not a regressor (2:00–3:15)

### We fitted the governing equations, not the data

```
dCA/dz = -k1·CA
dCB/dz =  k1·CA - k2·CB
dT/dz  =  a1·k1·CA + a2·k2·CB + U·(T_jacket - T)
```

**7 physical parameters. Fixed functional form. No architecture search** — the form came from
the chemistry, so there was nothing to search.

| Model | Protocol | RMSE |
|---|---|---|
| Best tree ensemble (ExtraTrees + physics features) | 10f-CV | **16.37** |
| **Physics ODE** | **10f-CV** | **6.36** |
| Physics ODE | training fit | 3.66 |

**Three methodology choices, each measured:**

- **Noise-averaged prediction.** We predict E[f(x+δ)] with σ = **1.67 K**, fixed *a priori*
  from an errors-in-variables fit — never tuned on the CV that judges it. It makes the
  *training* fit worse (3.656 → 3.764) and cross-validation better (6.358 → 5.928) on all
  three seeds. A blur that helps held-out error is evidence of input noise, not overfitting.
- **A 13% ExtraTrees component**, weight *derived* not tuned: from error correlation
  ρ = 0.070 between the two models, variance-optimal w\* = 0.887; we ship 0.87.
- **Validation.** Repeated 10-fold, plus cold-start folds refit from scratch (5.98) — two
  independent protocols agreeing. We also measured what our own protocol *cannot* price:
  leave-one-seed-out underprices hyper-parameter selection, because all seeds re-partition
  the same 150 rows. Row-level split-half puts that cost at **+0.371**.

**Why this approach for this problem:** 150 rows cannot support a flexible model at the yield
cliff, and a tree cannot extrapolate past its outermost split. An ODE can — and its
parameters are quantities a process engineer can check against known kinetics.

> **Notes (75 s).** The rubric asks "why *this* approach for *this* problem" — answer it in
> those terms: small n, sharp cliff, need for extrapolation and inspectability. The
> LOSO-limitation line is deliberate; volunteering a flaw in our own protocol is worth more
> than the 0.4 RMSE it costs us, and a judge who spots it first would cost us much more.

---

## Slide 4 — Results: what the score actually says (3:15–4:15)

### RMSE 11.0442 · MAE 3.2408 · R² 0.9061 → **rank 13 of 400**

**The two error metrics disagree, and that disagreement is the finding.**

### RMSE / MAE = **3.41**  (Gaussian errors give 1.25, Laplace 1.41)

A ratio that high cannot come from broadly mediocre prediction. Solving for an error
distribution that matches all three statistics on 50 rows:

| Rows missed badly | Their error | Error on the remaining ~47 rows |
|---|---|---|
| 2 | 55.0 | **1.09** |
| 3 | 45.0 | **0.57** |

**≈47 of 50 rows predicted to about 1 yield-point. Two or three cliff rows missed by ~50.**
R² 0.9061 implies a test-target spread of 36.0 against 38.3 in training — the test set is
distributed like the training set, which a two-sample classifier independently confirmed
(AUC 0.500, i.e. indistinguishable).

**Independent replication of the physics.** A separate blind refit — different integrator
(LSODA), different optimizer, no access to our code or notes:

| | Ours | Blind |
|---|---|---|
| E₁ | 43.161 | 43.156 |
| E₂ | 250.072 | 250.53 |
| a₁ | −11.7919 | −11.7961 |
| train RMSE | 3.6559 | 3.6554 |

**What we got right, and what we did not.** We pre-registered the failure *mechanism* — that
a handful of cliff rows would carry nearly all squared error — and the score confirms it. We
**underestimated its magnitude**: our own bootstrap over 50-row draws put RMSE in [2.94,
8.10], and 11.04 sits outside it. Our MAE prediction was accurate (observed 3.24, predicted
[1.74, 4.03]). Our 150 training rows never produced a miss larger than 40 points; the test
set produced two or three.

> **Notes (60 s).** Do not present the decomposition as an excuse — present it as the error
> analysis the rubric asks for (30 marks). The honest framing wins here: *we called the
> mechanism, we underestimated the magnitude.* If asked "so your CV was wrong" — yes, on the
> tail; it was right on the median and on MAE. That is what a 150-row training set buys you.

---

## Slide 5 — Where it fails, and how you would tell (4:15–5:00)

### If this started predicting badly tomorrow, three things could be wrong

| Source | Symptom |
|---|---|
| **Chemical** | Fouling raises U · catalyst degradation shifts k₁ · feed change moves CA₀ · operation outside 351.6–548.0 K |
| **Sensor / data** | Near the cliff a **2 K** temperature error becomes a ~20-point yield error. We measured a median input offset of **1.67 K**, and **12 of 150** rows resist any offset within ±25 K |
| **ML / model** | The cliff: **29% of rows carry 91%** of squared error. Our blend also lifts 6 dead-reactor predictions above 5% yield (max 11.5) — a real, documented cost |

### The diagnostic — and it only works because the model is mechanistic

**Refit the seven parameters on new data and read which one moved.**

| What moves | Diagnosis |
|---|---|
| **U** alone | heat-transfer fouling |
| **k₁** alone | catalyst degradation |
| **CA₀ path** only | feed composition change |
| Nothing moves, errors rise | sensor drift or an unmodelled phenomenon |

A black-box model cannot do this. It tells you the prediction is wrong; it cannot tell you
**which piece of the plant** changed. That is the practical case for recovering the reactor
instead of approximating it.

**Known limits, stated plainly:** 150 rows, one reactor geometry. Tube diameter is constant
across every row, so it folds into the fitted pre-exponentials — a different tube needs a
refit, not a rescale. And the cliff rows remain genuinely hard: we tested 8 independent
improvement strategies after submission and none cleared our adoption bar.

> **Notes (45 s).** The guidelines ask two questions almost verbatim: *what could cause
> incorrect predictions* and *how would you tell chemical from sensor from model*. Answer the
> second one directly — the parameter-attribution table is the strongest thing in this deck
> and is the natural closing note. End on the diagnostic, not on the RMSE.
