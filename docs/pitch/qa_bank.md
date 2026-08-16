# Q&A Bank — Team *Claude ke Chhatore*

One sentence and a number for each. Assume at least one reaction-engineering specialist on
the panel. Every figure carries its protocol.

---

## Reaction engineering

**1. Why is U constant across a 10x flow range? Turbulent heat transfer should scale as Re^0.8.**
We fitted the exponent rather than assume it: n = 0.8 costs **+11.40** train RMSE and the
95% interval is **[-0.03, 0.02]**, so the controlling thermal resistance is not on the
process side — most likely jacket-side or wall-limited — and flow enters only through
residence time.

**2. How do you know both reactions are first-order?**
We fitted the orders free rather than assume them: cross-validated RMSE gets **+0.072
worse** over 5 seeds (one seed -0.189) against a pre-registered 0.3 bar, so first-order is
measured, not assumed.

**3. The problem statement says "series-parallel". Where is your parallel path?**
We fitted one — adding ln_k3, E3 and a3 buys **+0.0019** train RMSE for three parameters
while 10-fold CV over 3 seeds gets **-0.352 worse** and unstable (5.41 / 8.13 / 6.59), so
the data supports the series network.

**4. Is your A→C rejection just a solver artifact?**
No — the A→C integrator is exact, reducing to the shipped model to **5.2e-12** yield-points
at k3 → 0 and agreeing with SciPy BDF to **9.1e-04** with k3 active.

**5. Why does concentration have almost no effect (corr = +0.009)?**
Not because the reactions are thermally neutral — we tested that and it costs **8.27 vs
3.66** — but because a1 = **-11.79** and a2 = **+11.34** have opposite signs and nearly
equal magnitude, giving adiabatic swings of -27 K then +26 K, a net of **-1 K**.

**6. Residence time shows correlation +0.061 with yield. Doesn't that make it irrelevant?**
The opposite — that is the signature of an interior optimum, where low tau leaves A
unconverted and high tau destroys B, so both tails are low-yield and the linear correlation
cancels.

**7. What is the single most useful number for a plant operator?**
The crossover temperature **449.9 K**, where k₂ overtakes k₁ — but state it as a joint
(T, τ) condition, not a ceiling: above it *every additional unit of residence time costs
selectivity*, so τ must be actively shortened rather than temperature simply capped.

**7b. So can the reactor run above 449.9 K?**
Yes, and our own data proves it — **49 training rows sit above the crossover and one reaches
75% yield**, because at short residence time B exits before it degrades; we predict **54.3%**
for a test row at 487.9 K with τ = 0.066.

**8. Is E1 = 43 kJ/mol physically plausible?**
Yes, it is a typical liquid-phase value; the informative result is the *ratio*
**E2/E1 = 5.79**, which is what makes the system selectivity-limited rather than
conversion-limited.

**9. Why is a2 so much less well determined than a1?**
Across cold-start folds a2 shifts 91% and ln_k2 61% while E1 and ln_k1 move only ~4%,
because k2 is either negligible or overwhelming across most sampled conditions — but the
governing ratio k2/k1 stays stable at 0.063–0.079.

**10. Did you consider axial dispersion?**
Yes — tanks-in-series at *fixed* parameters is worth only **~0.08** RMSE; the larger apparent
gain from refitting was the optimizer exploiting the coarse cascade's discretization error.

---

## Statistics and overfitting

**11. Seven parameters on 150 rows — prove this is not overfit.**
Cold-start folds, refit from scratch with no warm start, give held-out **5.98** against a
training fit of 3.66, and the warm-started 10-fold CV agrees at 6.36 — two independent
protocols landing in the same place.

**12. Why should we trust a CV number you also used to select hyper-parameters?**
We don't quote one — the headline **5.671** is leave-one-seed-out, where the policy is
chosen on two seeds and scored on the third, and sigma was fixed a priori at the
errors-in-variables median rather than tuned on that CV at all.

**12b. Does leave-one-seed-out actually price that selection?**
No, and we tested it rather than assuming — all three seeds re-partition the same 150 rows,
so choosing (w, cutoff) on 75 rows and scoring on the other 75 costs **+0.371 RMSE** and
loses 200/200 replicates, while LOSO prices the same choice at +0.02; **5.671 is a lower
bound**, though the policy sits on a plateau (full-data argmax (0.86, 60) scores 5.6546 vs
5.6547 shipped).

**13. Your in-sample optimum scores better than 5.671. Why not quote that?**
Because it is an argmax measured at its own optimum; the honest number is the one where the
scoring seed took no part in selection.

**14. How much of your final score is luck?**
A lot — bootstrapping our out-of-fold predictions over 50-row draws gives a 5th–95th
percentile of **[2.41, 9.07]**, and we would rather state that than present a point estimate
as a promise.

**15. What would change your mind about E2 = 250?**
A value outside the 95% interval **[233.75, 269.92]** fitting comparably well — and the
external audit's E2 ≈ 155 does not, sitting **+2.62** above optimum; an independent blind
refit landed at [234, 271].

**16. One of your cold folds scored 13.05. Isn't that a failure?**
It is genuine and reproducible at 3x the search budget — on those 135 rows a distinct
parameter basin fits better (train 2.56) and generalizes worse, and which basin a fit lands
in is set by **which rows are in the fold**: 10 of 100 folds sit there.

**17. Why not average over parameter uncertainty?**
We tested it: eight admissible starts spanning E2 = 210–265 refit inside a fold converge to
the same optimum within **2e-4**, so there is no posterior to average over — the apparent
spread was an artifact of pinning during profiling.

**18. Why did you profile with an F-test rather than the RMSE band your script prints?**
The 10%-RMSE band is a display heuristic and far too generous — it gives E2 ∈ [210, 320] —
whereas the F-test criterion gives **[233.75, 269.92]**, which is the statistical statement.

---

## The machine-learning component

**19. Why is there a random forest inside your physics model?**
It is a weak, decorrelated estimator at 13% weight that reduces ensemble variance by
averaging — correlation between its errors and the physics errors is **+0.070**, and it
scores **15.04** standalone against 6.14 for physics on the same rows.

**20. Isn't it correcting your ODE's bias?**
We could not demonstrate that at our pre-registered bar — 8 of 13 on the sign test against a
bar of 11/13 — though that is a failure to demonstrate rather than a demonstration of
absence, since at n = 13 the test only detects a very large effect and the magnitude is
directionally aligned (+3.29 supplied, +4.34 needed), so we claim the weaker interpretation.

**21. What does the blend cost you?**
On the 168 near-zero predictions slice RMSE goes **0.366 → 1.960** (a delta of -1.594,
costing 622.9 SSE) and it introduces **+1.083** of upward bias on the blended rows; we keep
it because the net is +0.27 across all three held-out seeds.

**21b. Your band numbers don't obviously add up to the total gain. Do they?**
They do, once you do it in squared error rather than RMSE — pooled over 3 seeds the ΔSSE is
low band +1321.3, mid band +765.9, dead band -622.9, summing to **+1464.3** with a closure
residual of 4.6e-13; we had earlier claimed the gain was "13 rows", which was wrong because
per-band RMSE deltas are not additive.

**22. If it's just variance reduction, why not bag the physics model instead?**
We tried — bagging recovers only **+0.13** of the +0.28 and made one seed worse; we
originally blamed the second likelihood basin, but only **7 of 2400** replicates ever cross
into it, so the rejection stands and that explanation does not.

**23. Would you drop the tree for a cleaner story?**
It would cost 0.27 RMSE against a sampling band of [2.41, 9.07], so it is a defensible call
either way — we kept it because the gain is measured on every held-out seed and sits on a
plateau, not a peak.

**24. Why not a neural network?**
150 rows, and the rubric explicitly penalizes brute force — but more concretely, boosting
measured *worst* of everything we tried, and no black box can state an operating limit of
449.9 K.

**25. Why did a residual ML corrector not help?**
Because there is little learnable structure left in our residual — three correctors over 5
seeds gained at most **+0.027** against a 0.3 bar; an external audit measuring against a much
weaker physics fit saw a large gain for exactly that reason.

---

## Data quality and the error budget

**26. How do you know the data is noisy rather than your model wrong?**
The decisive asymmetry is that smoothing over a 1.67 K input offset makes the *training* fit
worse (3.6559 → 3.7643) while improving 10-fold CV (6.3579 → 5.9279) on all three seeds — a
blur applied to a correct model with clean inputs would hurt CV, not help it.

**27. Could the residual just be a wrong E2 rather than input noise?**
Honestly, yes in part — a small error in E2 produces the same dY/dT signature, so the
sensitivity result is consistent with the noise story but not diagnostic of it.

**28. Have you accounted for the whole residual?**
No — **12 of 150 rows** cannot be reproduced by any temperature offset within ±25 K, and
against 10f-CV 6.36 the non-noise component is roughly 4.7, which is parameter-estimation
variance plus whatever those 12 rows are.

**29. Why is 29% of your error concentrated in so few rows?**
Not because they are intrinsically hard but because **dY/dT is large** there — the
least-sensitive fifth of rows sits at RMSE 0.200 and the most sensitive at 7.393, so the
error tracks sensitivity.

**30. Why did you predict [a cliff row] so differently from its nearest neighbour?**
Because near the cliff a 2 K difference in operating temperature is a 20-point difference in
yield — the neighbours are close in input space and far apart in outcome, which is exactly
what an interior optimum with a steep flank produces.

**30b. Does your model ever predict a meaningful yield on a completely dead reactor?**
Yes — the blend lifts 15 of 111 dead-row predictions above 1.0 and **6 above 5.0, maximum
11.499** (4.2% of total squared error), where pure physics contributes 0.0% there with a
maximum of 0.102; we left it because the perfect repair is worth only **+0.119** and would
be a fourth tuned rule.

**31. You lifted a true-zero training row off zero. Isn't that a defect?**
Row 97's truth is **0.282**, not 0.000, which marks it an *edge* row rather than an interior
one — interior dead rows are exactly 0.000 in 15 of 15 cases — so the lift is smoothing
correctly expressing cliff uncertainty, at a cost of ~5 squared-error units against ~616
recovered.

**32. Did you remove any outliers?**
No — the extreme rows are real physics regimes and carry the location of the yield cliff,
which is the part of the surface that decides the score.

---

## Deployment and scale

**33. How fast is it?**
**24.8 ms** for all 150 rows at 512 substeps, against 5573 ms for a SciPy BDF reference solve
of the same system — a measured **225x**, agreeing to 0.017 yield-points.

**34. Is that 225x faster than the original plant simulation?**
No, and we are careful about this — the problem statement never publishes a runtime, so we
compare only against a reference solver we timed ourselves.

**35. What happens outside the training envelope?**
The ODE still integrates and returns physics, where a tree would clamp to its outermost
split — but our error bars widen, and the honest statement is that the parameters were
identified on 351.6–548.0 K.

**36. How does this scale to a real plant?**
The kinetics transfer; the geometry does not — tube diameter is constant across all 150 rows,
so it folds into the fitted pre-exponentials and into U, meaning a different tube needs a
refit rather than a rescale.

**37. What would you do with more data?**
Sample the cliff — 29% of rows carry 91% of the error, so rows near the crossover are worth
several times a row in the well-determined majority.

**38. How would a process engineer audit this model?**
By checking E1 = 43.16 and E2 = 250.07 kJ/mol against known kinetics for this chemistry and
rejecting it on chemical grounds — which is a kind of review no validation curve supports.

---

## Process and integrity

**39. What did you get wrong?**
Six things, on our own slide — most sharply, we once claimed the tree corrected the ODE's
bias (falsified by the sign test) and once ruled out a parallel path with an argument that
only bounded k3 at a single row's 383 K while the data spans to 548 K.

**40. You said the 99.97% argument reached the right answer. Why call it an error?**
Because it was a lucky assertion rather than a measurement — the conclusion survived, but the
reasoning did not support it, and we replaced it with a fit.

**41. You found a bug that reversed a conclusion you disliked. How do we know it was a real bug?**
The fix is defensible on its own terms — the original code silently dropped all 41 rows the
model already fit well, inflating the spread — but the search for it was motivated by
disliking the result, and we think that is worth disclosing rather than hiding.

**42. How do you know your deck matches your submission?**
`scripts/audit_pitch.py` reads every number in these documents live from `artifacts/`,
verifies the submission's sha256 is `c68e0e748e4928f2`, checks the policy triple agrees
across four files, and fails if any retracted number reappears.
