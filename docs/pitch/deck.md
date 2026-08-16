# Recovering the Reactor
### Team *Claude ke Chhatore* — Phase 2 pitch

> **Source of truth.** Every number in this deck is declared in `scripts/audit_pitch.py`
> and checked against `artifacts/` on every run. If a number here disagrees with the
> repository, the audit fails. Run `python scripts/audit_pitch.py` to verify.

---

## Slide 1 — The trade-off that decides everything

**Two reactions compete for the same molecule.**

```
A --k1--> B --k2--> C
          ^product   ^waste
```

B is the product **and** the feedstock for the waste reaction. Both rates are Arrhenius, so
heat accelerates both — but not equally:

| | Activation energy |
|---|---|
| Desired, A → B | **E1 = 43.16 kJ/mol** |
| Waste, B → C | **E2 = 250.07 kJ/mol** |

### **E2 / E1 = 5.79**

Heating accelerates the destruction of B almost six times as steeply as its formation. The
two rate constants cross at **449.9 K**. Below it, the reactor makes B faster than it
destroys it. Above it, the reverse — and yield collapses.

**That single ratio explains the whole dataset.** It is why the correlation between jacket
temperature and yield is *negative* (−0.498). It is why a quarter of the training rows are
exactly zero. It is the reason there is an optimum at all.

> **Presenter notes.** Open here, not on the modelling. The rubric scores process insight,
> so lead with the physical statement and let the ML follow from it. The crossover is
> computed from the fitted parameters, not read off a chart — it sits at 449.9 K, inside
> the training temperature envelope of 351.6–548.0 K, so it is interpolated, not
> extrapolated. If asked how a ratio of activation energies can "explain a dataset", the
> answer is that it sets the sign of dY/dT, and the sign of dY/dT is what makes the yield
> surface a ridge rather than a slope.

---

## Slide 2 — How we measure, stated once

**One submission. No leaderboard. Every decision made against cross-validation.**

We report **leave-one-seed-out (LOSO)**: the prediction policy is chosen on two random seeds
and scored on the third.

**And we will tell you its limitation before you ask.** All three seeds re-partition the
*same 150 rows*, so the "held-out" seed has already seen every row. We measured what that
costs: choosing the blend weight and cutoff on 75 rows and scoring on the other 75 costs
**+0.371 RMSE** against just using our fixed values, and selection loses in **200 of 200**
replicates. LOSO prices the same choice at +0.02. So **5.671 is a lower bound, not an
unbiased estimate** — LOSO prices fold-partition noise, not selection.

What rescues the *policy*, as opposed to the number: every search lands at a weight between
0.85 and 0.89 with the same cutoff of 60, and the full-data optimum is (0.86, 60) scoring
5.6546 against 5.6547 for what we ship. We are on a plateau, not a peak.

| Model | Protocol | RMSE |
|---|---|---|
| Best tree ensemble (ExtraTrees + physics features) | 10f-CV | **16.37** |
| Physics ODE, raw | 10f-CV | 6.36 |
| Previous policy | LOSO | **6.022** |
| **SHIPPED — sigma 1.67 K, w 0.87, cutoff 60** | **LOSO** | **5.671** |

Every other figure in this deck carries its protocol inline. They are **not** comparable to
each other.

**And the honest caveat:** a 50-row test set is small. Bootstrapping our out-of-fold
predictions gives a 5th–95th percentile of **[2.41, 9.07]**. A large part of our final
score is luck, and we would rather say so than pretend a point estimate is a promise.

> **Presenter notes.** Put this second, before any result, so no number later needs
> defending on protocol grounds. The bootstrap band pre-empts "you got lucky / unlucky".
> **Deliver the LOSO limitation deliberately** — we found it ourselves, late, by measuring
> the selection cost at row level rather than seed level, and a careful judge would otherwise
> find it for us. The distinction to hold onto: the *number* is optimistic, the *policy* sits
> on a plateau and is not overfit. If pushed on what we would do differently — price any new
> policy knob with row-level split-half or leave-one-row-out, because adding CV seeds
> re-partitions the same 150 rows and can never price selection.

---

## Slide 3 — We recovered the reactor rather than approximating it

We did not fit a regressor to the target. We wrote down the governing equations and fitted
their physical parameters.

```
dCA/dz = -k1*CA
dCB/dz =  k1*CA - k2*CB
dT/dz  =  a1*k1*CA + a2*k2*CB + U*(T_jacket - T)
```

**Seven parameters. A fixed functional form. No architecture search.**

| | Our ODE | A tree ensemble |
|---|---|---|
| Free parameters | **7** | thousands of effective splits |
| Extrapolates? | yes, it is the mechanism | no, bounded by outermost split |
| Parameters mean something | activation energies, heats of reaction | node thresholds |
| 10f-CV RMSE | 6.36 | 16.37 |

**One measurement makes the point sharper than the table does.** A quarter of the rows are
dead reactors, so "is this reactor dead?" is a natural classification task — the kind a
discriminative model should own. Pooled out-of-fold AUC:

| Model | AUC |
|---|---|
| **Physics ODE prediction** | **0.9587** |
| ExtraTrees | 0.9505 |
| Logistic regression | 0.9253 |
| Gradient boosting | 0.9138 |

Seven physical parameters beat every classifier we fitted, at the one task classifiers are
supposed to be good at — because the ODE knows *why* a reactor is dead.

Two implementation choices earned their place:

- **Arrhenius reparameterized about 430 K.** Fitting ln A and E directly correlates them
  above 0.999 and turns the objective into a long narrow valley.
- **An operator-splitting integrator** that advances the mass balance *analytically* and
  the energy balance with the exact linear solution. Both halves are unconditionally
  stable. Fixed-step RK4 blew up on the thermal-runaway rows; this does not.

> **Presenter notes.** The parameter-count contrast is the Innovation criterion in one
> table. Do not oversell "no architecture search" — the point is that the functional form
> came from chemistry, so there was nothing to search. If a judge pushes on the integrator,
> the honest framing is that we needed thousands of evaluations for a global fit, and 150
> separate solve_ivp calls per evaluation was unaffordable; the analytic step made
> multistart possible. Its accuracy is verified against SciPy BDF, not assumed.

---

## Slide 4 — The operating map

**Yield is a ridge, not a slope.** Too cold or too fast, A never converts. Too hot or too
slow, B is destroyed. The optimum is a narrow band between them.

This produces a result that looks like a null and is not:

### corr(log tau, yield) = +0.061

**Residence time is the single most important derived quantity in the problem** — and its
linear correlation with yield is essentially zero. That is the signature of an **interior
optimum**: both tails are low-yield, so the correlation cancels.

A model selecting features by correlation would discard residence time entirely. This is
the clearest example of why we did not let a feature-importance ranking drive the model.

> **Presenter notes.** This slide exists to demonstrate reading a diagnostic correctly. It
> is also a trap we watched others fall into: the same logic applies to concentration,
> which is the next slide. If a judge asks what we would have done without the physics
> model, the answer is that a partial-dependence plot would have shown the ridge — but we
> would not have known *why*, and could not have quoted a crossover temperature a plant
> engineer can act on.

---

## Slide 5 — Why concentration does not matter, and why the obvious answer is wrong

### corr(concentration, yield) = +0.009

Concentration is a knob the operator controls, and it does essentially nothing. Why?

**The tempting answer:** "both reactions are first-order, so CA0 cancels." True for the
isothermal yield expression — but incomplete. A richer feed releases more reaction heat,
which should change yield through the thermal path.

**The next tempting answer:** "so the reactions must be thermally neutral." **We tested
this and it is false.** Forcing a1 = a2 = 0 costs train RMSE **8.27** against 3.66. The heat
terms are real and large.

**What actually happens:**

| | |
|---|---|
| a1 (A → B) | **-11.79** K·L/mol |
| a2 (B → C) | **+11.34** K·L/mol |

Opposite signs, near-equal magnitude. At mean feed concentration the adiabatic swings are
**-27 K then +26 K — a net of -1 K.** The two heat effects very nearly cancel.

That is the defensible answer, and we only have it because we fitted the mechanism.

> **Presenter notes.** This is the strongest process-insight slide; do not rush it. The
> structure — obvious answer, better answer, both wrong, measured answer — is the whole
> pitch in miniature. Note that first-order kinetics is itself something we measured rather
> than assumed: see slide 6. If asked whether the cancellation is a coincidence, say we do
> not know; it is what the data supports, and a1 is determined tightly enough (95% CI
> [-12.12, -11.49]) that the cancellation is not an artifact of a loose fit.

---

## Slide 6 — What we tried to break

Every entry below was **fitted and measured**, not argued away. Costs carry their protocol.

| Hypothesis | Result | Protocol |
|---|---|---|
| Thermally neutral reactions (a1 = a2 = 0) | 8.27 vs 3.66 | train |
| a1 = 0 alone — testing our *own* concentration mechanism | **+4.54** | train |
| **Parallel A → C path** | train **+0.0019** for 3 extra params; CV **-0.352** worse | train / 10f-CV, 3 seeds |
| Flow-dependent jacket U, (Q/Qref)^n | n = 0.8 costs **+11.40**; 95% CI [-0.03, 0.02] | train / profile |
| Free reaction orders n1, n2 | +0.072 worse, one seed -0.189 | 10f-CV, 5 seeds |
| Axial dispersion (tanks-in-series) | ~0.08 at fixed parameters | train |
| Residual ML corrector on our residuals | best +0.027 against a 0.3 bar | 10f-CV, 5 seeds |
| E2 = 155 kJ/mol (an external audit's value) | **+2.62** | train |

**The parallel path deserves a sentence.** The problem statement calls this a
*"series-parallel reaction network"* while listing only A → B → C. So we fitted the parallel
path rather than assume either reading. Three extra parameters buy **0.002** on training
error and **lose 0.35** on cross-validation, swinging 5.41 / 8.13 / 6.59 across seeds. That
is what fitting noise looks like. **The data supports the series network.**

**The flow result is a positive finding, not just a rejection.** Turbulent heat transfer
would give n ≈ 0.8. We measure n ≈ 0, with a 95% interval excluding even 0.03. So the
controlling thermal resistance is **not on the process side** — flow enters this reactor
only through residence time. That is actionable: increasing flow will not improve jacket
duty.

> **Presenter notes.** This is the Robustness criterion. Lead with the A→C row, because the
> problem statement's own wording invites the question and most teams will not have tested
> it. Emphasise that our A→C integrator is *exact*, not an approximation — it reduces to the
> shipped model to 5e-12 at k3→0 and matches SciPy BDF to 9e-04 with k3 active — so the
> rejection is not an artifact of a sloppy solver. If challenged on the 0.3 bar, it is
> pre-registered: we set it before running, at roughly one seed-to-seed standard deviation.

---

## Slide 7 — Independent replication

We ran a **blind refit**: a separate agent, given only the data and the problem statement,
with **no access to our code, parameters, or notes**. It used a different integrator (SciPy
LSODA) and a different optimizer.

| Parameter | Ours | Blind refit |
|---|---|---|
| E1 (kJ/mol) | 43.161 | 43.156 |
| E2 (kJ/mol) | 250.072 | 250.53 |
| a1 | -11.7919 | -11.7961 |
| a2 | +11.3409 | +11.3212 |
| U | 3.2552 | 3.2554 |
| train RMSE | 3.6559 | 3.6554 |

**Agreement to roughly five significant figures**, from a different numerical stack.

It independently reproduced the falsification too: forcing a1 = a2 = 0 gives **8.2679**
against our **8.2683** — four decimal places.

And the E2 confidence interval: ours **[233.75, 269.92]**, the blind refit's **[234, 271]**.

> **Presenter notes.** This is the answer to "how do we know your optimizer did not just
> find a convenient local minimum". Two independent implementations landing on the same
> parameters is much stronger than any single-run diagnostic. Be precise about what "blind"
> means — the agent could not read our repository, including the README and notes that state
> E1 and E2. If asked why E2 agrees least well (250.07 vs 250.53), that is expected: E2 is
> our least tightly determined kinetic parameter, and both values sit comfortably inside
> both confidence intervals.

---

## Slide 8 — Where the error actually lives

Grouping training rows by how much the *data itself* disagrees locally — the spread of
yields among each row's six nearest neighbours in (log tau, T_in, T_jacket):

| Neighbour spread | n | OOF RMSE | Share of squared error |
|---|---|---|---|
| 0–20 | 47 | **0.652** | 0.4% |
| 80–101 | 44 | **10.410** | **91.0%** |

### 29% of rows carry 91% of the error.

**And we can say why.** It is not that those rows are intrinsically hard — it is that
**dY/dT is large there**. Grouping instead by sensitivity, the least-sensitive fifth of rows
sits at RMSE **0.200** and the most-sensitive fifth at **7.393**. The error tracks
sensitivity, which is exactly what a small input perturbation would produce.

**The consequence for effort allocation:** further modelling of the well-determined majority
cannot move the score. The result is decided by how the cliff-edge rows fall.

> **Presenter notes.** This is the slide that justifies every "we stopped here" decision
> later. The distinction between "hard rows" and "sensitive rows" is the insight: sensitive
> rows are where a 2 K input error becomes a 20-point yield error, which is a property of
> the reactor near its cliff, not a deficiency of the model. It also explains why the tree
> is blended in only at low predicted yields — that is where the cliff is.

---

## Slide 9 — The information limit, stated at the strength the evidence supports

We believe we are near the limit of what these 150 rows determine. Here is the evidence,
and here is where it stops.

**What we measured:**

- **Residual scales with input sensitivity** — least-sensitive quintile **0.200**,
  most-sensitive **7.393**. This rules out *output* noise, which would be uniform.
- **No residual structure.** A scan over 104 features and pairwise products finds nothing
  (largest |r| = 0.137, expected false positives ≈ 0.2).
- **A median temperature offset of 1.67 K reproduces 138 of 150 rows** exactly, and those
  offsets are structureless (largest feature correlation 0.206).
- **Smoothing over that offset degrades training error while improving CV** — train
  3.6559 → 3.7643, 10f-CV 6.3579 → 5.9279, better on all three seeds. **A blur applied to a
  correct model with clean inputs would hurt CV, not help it.** This is the strongest
  single result we have.

**What we will not claim:**

- **There is one locality where our ODE is measurably off.** In the 0.5–10% predicted-yield
  band the physics **misses low** — band RMSE 11.21 against 5.94 overall. State the shape
  carefully: the mean shortfall is **+3.99** but the median only **+0.70**, so it is not a
  uniform four-point offset but a handful of badly-missed cliff rows, with |need| reaching
  **38 yield-points**. This holds whether or not the tree is in the model, and it is the same
  29%-of-rows/91%-of-error concentration seen on the previous slide.
- **12 of 150 rows cannot be reproduced by any offset within ±25 K.** Something else is
  going on in those rows and we have not identified it.
- We do **not** claim the residual is fully accounted for. Against 10f-CV 6.36, the
  non-noise component is roughly 4.7 — parameter-estimation variance plus those 12 rows.
- We do **not** cite residual unbiasedness as evidence. Least squares forces the residual
  orthogonal to df/dtheta, so a small mean residual is a first-order condition of the
  optimizer, not a fact about noise. We used to present it as a pillar. It is not one.
- **A caveat against our own conclusion:** a small error in E2 would produce the same
  dY/dT signature as input noise. The sensitivity result is *consistent with* the noise
  story, not diagnostic of it.

> **Presenter notes.** Deliver the second half as deliberately as the first. Volunteering
> the 12 unexplained rows and the E2-mimicry caveat is more persuasive than a clean story,
> and it inoculates against a judge finding them. The load-bearing argument is the
> train-degrades/CV-improves asymmetry — that one does not depend on the mechanism being
> input noise, only on the observation that smoothing helps out of sample.

---

## Slide 10 — The random forest inside our physics model, declared

**There is a 13% weight on an ExtraTrees model in our submission. We are telling you before
you find it.**

**We could not demonstrate bias correction at our pre-registered bar** — sign test 8 of 13
(62%, p = 0.581) against a bar of ≥ 11/13. The mean correction *is* directionally aligned
(**+3.29** supplied against **+4.34** needed, same sign, ~76% of what is required), but at
n = 13 we cannot separate that from variance reduction. **So we claim the weaker
interpretation.**

| Test | Result |
|---|---|
| Sign agreement in the gain-carrying band (0.5 < p ≤ 10) | **8** of **13** (62%, p = **0.581**) |
| Bar we pre-registered | ≥ 11/13 |
| Correlation between physics error and tree error | **+0.070** |
| Tree standalone RMSE on those rows | **15.04** vs 6.14 for physics |

A weak, decorrelated estimator improves an ensemble by averaging even when it is far worse
standalone — and that mechanism needs no bias claim to hold.

### Where the gain comes from — reconciled in squared error

RMSE is not additive; squared error is. Pooled over 3 seeds × 150 rows:

| Band | n | RMSE pre → post | ΔSSE | % of gain |
|---|---|---|---|---|
| dead, p ≤ 0.5 | 168 | 0.366 → **1.960** | **-622.9** | **-42.5%** |
| low, 0.5 < p ≤ 10 | 39 | 11.210 → 9.580 | +1321.3 | +90.2% |
| mid, 10 < p ≤ 60 | 86 | 9.441 → 8.957 | +765.9 | +52.3% |
| above cutoff | 157 | unchanged | 0.0 | 0.0% |
| **TOTAL** | **450** | **5.935 → 5.655** | **+1464.3** | **100%** |

**This corrects an earlier claim of ours** that the whole gain came from 13 rows. It does
not — the mid band contributes 52% of it. And the gain is **tail-dominated**: the top 3 of
the 39 low-band rows supply 77% of that band's improvement, with pre-blend errors of 21–38
yield-points against a band median |need| of just 1.46.

**The cost, stated as a level not just a delta:** on the 168 near-zero predictions the slice
RMSE goes **0.366 → 1.960**, a delta of **-1.594** and a cost of 622.9 SSE. Blending also
introduces **+1.083** of upward bias on the blended rows.

**And the specific version of that cost you should hear from us, not find yourselves.** The
pure physics model contributes **0.0%** of its squared error on the 37 truly-dead rows — its
largest prediction there is 0.102. The blend lifts **15 of 111** dead-row predictions above
1.0, **6 above 5.0, with a maximum of 11.499**, which is 4.2% of total squared error. So yes:
on one dead reactor our shipped model predicts about 11% yield. We left it because the perfect
repair is worth only **+0.119** and fixing it would be a fourth tuned rule — but it is a real
cost of buying variance with a tree. We keep it because the net is
**+0.27** across all three held-out seeds, the weight sits on a plateau rather than a peak,
and we tested the principled alternative — **bagging the physics fit** — which recovered only
+0.13 and destabilised one seed.

**And we got the reason wrong the first time.** We used to say bootstrap replicates land in a
second optimum on the likelihood surface. Measured: basin membership is set by *which rows are
in the fold*, not by the resample — 10 of 100 folds sit in the second basin, but only **7 of
2400 replicates** ever cross between them (0.3%). The rejection stands at +0.13 against a 0.3
bar; the mechanism we attached to it did not survive being checked.

> **Presenter notes.** Deliver deliberately; do not get discovered by it. Be careful with the
> claim structure here — we say we *could not demonstrate* bias correction, not that it is
> absent, because at n = 13 the sign test only detects a very large effect and the magnitude
> evidence actually leans the other way. Claiming the weaker interpretation is the defensible
> position, and a statistician on the panel will notice if we overstate it. The SSE table is
> there because the per-band RMSE numbers do not sum and reading them as contributions is
> what led us to claim the gain was 13 rows; it is 90% low band, 52% mid band, −43% dead
> band. If a judge says "that is an ML crutch in a physics pitch", agree it is the least
> elegant part of the model and point at the pre-registered bar we tested it against.

---

## Slide 11 — What we got wrong

**Six claims we made, and what falsified them.** No other team will bring this slide.

| We claimed | What falsified it | What we say now |
|---|---|---|
| "The tree corrects the ODE's systematic bias" | Sign test: no significant alignment | Variance reduction from a decorrelated component |
| "Residuals are unbiased, therefore noise" | Least squares *forces* orthogonality to df/dtheta | Near-vacuous. Demoted, not quoted |
| "Yields reach 99.97%, so a parallel path is excluded" | Only bounds k3 **at that row's 383 K**; data spans to 548 K | Replaced with a fitted measurement |
| "ln k3 goes to its bound, CV delta 0.0003" | Never measured. When measured: k3 is seed-dependent, delta is **-0.352** | The fit rejects A→C on generalization |
| First errors-in-variables run showed structure | A bug silently dropped all 41 well-fit rows | Fixed: 138/150 explained, structureless |
| Notebook hardcoded sigma = 2.5 after the policy moved to 1.67 | Re-execution produced a different submission hash | Policy now read from `blend.json`; both paths verified identical |
| "LOSO prices hyper-parameter selection" | Row-level split-half: selecting costs **+0.371**, losing 200/200. LOSO says +0.02 | LOSO prices fold-partition noise; 5.671 is a lower bound |
| "Bagging failed because replicates land in a second basin" | Basin membership is set by the fold, not the resample — **7 of 2400** replicates cross | Rejection stands at +0.13; the mechanism was wrong |

**Two of these deserve emphasis:**

**The 99.97% argument reached the right conclusion by invalid reasoning.** A parallel path
*is* excluded — but not for the reason we gave. That is the difference between a measurement
and a lucky assertion, and we count it as an error even though the answer survived.

**The EIV bug was found only after we saw a result we disliked.** The fix is correct on its
merits and we would defend it independently. But the search for it was motivated, and that
is worth disclosing rather than presenting the corrected result as if it arrived clean.

> **Presenter notes.** This is the highest-risk and highest-reward slide. Deliver it without
> hedging or apology — the framing is "here is our error-correction process working", not
> "here are our failures". The EIV disclosure in particular is the kind of thing a careful
> judge will respect precisely because we did not have to say it. Do not let it run long;
> one sentence each, then move.

---

## Slide 12 — Result, and what it would take to deploy

### Shipped: LOSO **5.671**, against **16.37** for the best tree ensemble

Policy: physics predictions noise-averaged over sigma = **1.67** K, blended **0.87** / 0.13
with the tree below a predicted yield of **60**, pure physics above.

**Speed — measured, not asserted:**

| | 150 rows |
|---|---|
| Our surrogate (512 substeps) | **24.8 ms** |
| SciPy BDF reference solve | 5573 ms |
| **Speed-up** | **225x** |

...and they agree to 0.017 yield-points. The original BVP simulation is, in the problem
statement's own words, *"computationally expensive and too slow for real-time plant
optimization"* — we do not quote a runtime for it, because none is published.

**What deployment actually needs:**

- **It extrapolates**, because it is the mechanism. A tree cannot predict past its outermost
  split; an ODE integrates whatever conditions you hand it. Outside the training envelope
  our error bars widen, but the model still returns physics rather than a clamped constant.
- **It is inspectable.** A plant engineer can check E1 and E2 against known kinetics and
  reject the model on chemistry, not on a validation curve.
- **It states its own operating condition — and the condition is joint in (T, tau).**
  449.9 K is where the *rate constants* cross, not a ceiling on temperature. Yield depends on
  k·tau, so a hot reactor with a short residence time still performs: **49 training rows sit
  above the crossover and one reaches 75% yield**, and we predict **54.3%** for a test row at
  487.9 K with tau = 0.066. The correct statement is that **above 449.9 K every additional
  unit of residence time costs selectivity**, because k2 now outruns k1 — so past the
  crossover, tau must be actively shortened rather than temperature simply capped.
- **Honest limit:** 150 rows, one reactor geometry. Tube diameter is constant across all
  rows, so it folds into the fitted pre-exponentials and into U. A different tube requires a
  refit, not a retune.

> **Presenter notes.** Close on the operating condition, not the RMSE — the RMSE wins Phase 1
> and the process insight wins Phase 2. **State it as (T, tau), never as a temperature
> ceiling:** a reaction engineer will immediately point out that at short residence time B
> exits before it degrades, and our own training data contains a row above the crossover
> yielding 75%. Getting this wrong on the closing slide would undo the credibility of
> everything before it. The 225x is measured end-to-end against a solver we verify against,
> so it is defensible as like-for-like; be careful not to imply it is 225x faster than the
> organisers' original simulation, which we have never timed. If asked about scale-up, the
> diameter caveat is the honest answer and volunteering it beats being asked.
