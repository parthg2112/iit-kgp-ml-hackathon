# Study Guide: The Problem and Our Solution

**Team-internal. Not for judges.** Written to be studied once and referenced afterwards, so
that any of us can defend any part of the work without reading from a slide.

Assumed starting point: high-school chemical kinetics (rate laws, Arrhenius) and RMSE as a
formula. Everything beyond that is either built up here or defined in one line and linked.

> Blocks like this one are coaching notes. They give what to *say* under questioning, and
> would be stripped before this document was shown to anyone outside the team.

---

## 1. The problem, from the ground up

### 1.1 What the reactor physically is

The system is a [plug-flow reactor](https://en.wikipedia.org/wiki/Plug_flow_reactor_model),
which is a tube. Liquid enters at one end, reacts as it travels, and exits at the other.
"Plug flow" means every molecule spends the same time inside with no back-mixing, so the
fluid moves like a series of independent plugs sliding down the pipe.

The tube sits inside a **jacket**, an outer shell carrying fluid at a controlled temperature,
and heat crosses the wall between them. The reactor is **non-isothermal**, meaning the
temperature varies along the tube. It varies for two reasons: the reactions themselves
release or absorb heat, and the jacket pushes heat in or pulls it out.

Three quantities therefore change as the fluid moves down the pipe. How much A is left, how
much B has formed, and how hot the fluid is. Each of the three affects the other two.

### 1.2 The reaction, and why it creates a trade-off

```
A --k1--> B --k2--> C
          ^          ^
       product     waste
```

This is a **series** (or consecutive) reaction. A becomes B, but B does not stop there,
because B then becomes C. The product we want is also the reactant for the reaction that
destroys it.

The trade-off follows from first-order kinetics alone. Both steps are
[first-order](https://en.wikipedia.org/wiki/Rate_equation#First-order_reactions), so each
rate is proportional to the concentration of its own reactant. Starting from pure A:

- Early on there is a large amount of A and almost no B. The first reaction runs quickly
  while the second runs slowly, because there is barely any B for it to consume. **B
  accumulates.**
- Later, A is largely consumed, so B production slows. Meanwhile B has built up, so its
  destruction speeds up. **B depletes.**

Plotting B against time therefore produces a hump: it rises, peaks, and falls. That peak is
an **interior optimum**. Run the reactor too briefly and unconverted A leaves the tube; run
it too long and the product has already become waste. This shape governs the entire problem.

Temperature then determines where the peak sits. Both rate constants follow
[Arrhenius](https://en.wikipedia.org/wiki/Arrhenius_equation) behaviour, `k = A·exp(-E/RT)`,
so heating accelerates both. It does not accelerate them equally, because the reaction with
the larger activation energy is the more temperature-sensitive of the two. The values we
recovered:

| Reaction | Activation energy |
|---|---|
| Desired, A to B | **E₁ = 43.16 kJ/mol** |
| Waste, B to C | **E₂ = 250.07 kJ/mol** |

The ratio **E₂/E₁ = 5.79** is the single most important number in the problem. Heating
accelerates the destruction of B roughly six times more steeply than its formation. Below
some temperature k₁ exceeds k₂, and the reactor produces B faster than it loses it. Above
that temperature the ordering reverses. The two rate constants cross at **449.9 K**, which
sits inside the observed range of 351.6 to 548.0 K.

> If asked which single number matters most, give E₂/E₁ = 5.79 and the crossover at 449.9 K
> that it implies. Everything else in the dataset follows from it.

### 1.3 The five inputs and what each one controls

| Input | Range in our data | Correlation with yield | What it controls |
|---|---|---|---|
| `jacket_temperature_K` | 354.0 to 548.0 | **-0.498** | heat pushed in or pulled out through the wall |
| `inlet_temperature_K` | 351.6 to 498.6 | -0.405 | how hot the feed arrives |
| `flow_rate_L_min` | 5.4 to 79.0 | +0.038 | residence time; faster flow gives less time |
| `length_m` | 2.3 to 25.0 | +0.080 | residence time; a longer tube gives more time |
| `concentration_mol_L` | 0.52 to 3.97 | **+0.009** | how strong the feed is |

Although there are five inputs, only three physical quantities are being set. Two
temperatures, and, through flow rate and length acting together, a **residence time**
τ = length / flow. Concentration is the fifth knob, and it does almost nothing. Section 1.5
explains why.

> Both temperature correlations are negative: a hotter reactor gives lower yield. This is
> direct evidence that E₂ exceeds E₁, available before fitting anything.

### 1.4 What yield is, and why it can be zero

Yield is defined as 100 × (moles of B leaving) / (moles of A entering). It is a percentage of
the theoretical maximum, so if every molecule of A became B and stayed B, the yield would be
100.

Across our 150 training rows, **37 rows (24.7%) are exactly 0.000**, 57 rows (38%) fall below
1.0, and 18 rows (12%) exceed 90. The target is therefore
[zero-inflated](https://en.wikipedia.org/wiki/Zero-inflated_model) and bimodal, clustering at
the two extremes with relatively little in between.

One property of those zero rows makes the rest of the problem intelligible, and it runs
against intuition. The natural assumption is that a zero-yield reactor never got going: too
cold, too fast, no reaction. The data says otherwise.

**Of the 37 exactly-zero rows, 31 are hot (T_avg above 449.9 K). Only 6 are cold.**

A dead reactor is therefore usually one in which B was produced and then completely
destroyed. The reactor ran hot, passed the crossover, and k₂ consumed everything. This is
why zero yields and 90%-plus yields sit close together in input space: a modest temperature
change moves a reactor from one to the other.

### 1.5 Two variables that look unimportant and are not

**Concentration**, with a correlation of +0.009. The tempting explanation is that both
reactions are first-order, so the initial concentration cancels out of the yield expression.
That holds for an *isothermal* reactor but is incomplete here, because a richer feed releases
more reaction heat, which changes the temperature, which changes both rate constants.

The next tempting explanation is that the reactions must therefore be thermally neutral. We
tested this and it is false: forcing both heat terms to zero costs train RMSE 8.27 against
3.66. The heat effects are large, but they **oppose each other**. The fitted coefficients are
a₁ = -11.79 and a₂ = +11.34 K·L/mol, so at mean feed concentration a swing of -27 K is
followed by a swing of +26 K, for a net of **-1 K**.

**Residence time**, with a correlation of +0.076, is arguably the most important derived
quantity in the problem despite that near-zero figure. An interior optimum produces exactly
this signature: too short is bad and too long is bad, so the correlation cancels between the
two failing tails. A model that selected features by correlation would discard residence time
altogether.

> Both variables are traps. If a judge asks why we did not use feature importance to select
> inputs, the answer is that the two most physically important variables both have near-zero
> linear correlation with the target.

### 1.6 Why the problem is hard

1. **The cliff.** The response between "dead" and "high yield" is close to a step function.
   Small input changes near that boundary produce very large output changes.
2. **Competing reactions.** Neither temperature nor residence time can be optimised alone,
   since raising either one helps the desired reaction and the waste reaction at once.
3. **Only 150 rows.** Any flexible model has more effective freedom than the data supports,
   and the shortage is worst near the cliff, where the rows are sparsest.
4. **One submission, no leaderboard.** There is no feedback loop, so every decision must be
   justified before any score is ever seen.

---

## 2. How to think about this problem, and where ML fits

### 2.1 Three possible approaches

**Pure machine learning.** Treat the task as tabular regression, with five features in and
one number out, and fit a [gradient-boosted tree](https://en.wikipedia.org/wiki/Gradient_boosting)
or [random forest](https://en.wikipedia.org/wiki/Random_forest). This is fast and requires no
chemistry. Our measured result on this route was **16.37** RMSE under 10-fold
cross-validation for the best tree ensemble. Trees split on axis-aligned thresholds, so they
approximate a smooth curved surface with staircases, and they cannot
[extrapolate](https://en.wikipedia.org/wiki/Extrapolation) past their outermost split at all.

**Pure physics.** Write the differential equations that govern the reactor, then fit their
physical parameters to the data. The functional form comes from chemistry, so there is no
architecture to search over, only seven numbers to estimate. Our measured result was **6.36**
RMSE under the same protocol, roughly 2.6 times better than the best tree.

**Hybrid.** Use the physics as the model, and let a small machine-learning component sit
alongside it to cover what the physics does not capture.

### 2.2 Where machine learning can actually act

The ODE has physical parameters rather than features, so feature engineering cannot improve
it. There is nowhere to put a feature. Machine learning can therefore affect only the second
component, the tree blended in at 13% weight.

This reframes the question precisely: does a better second model justify giving it more
weight? We answered it with [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning)
algebra rather than by guessing. For two models with error magnitudes s₁ and s₂ and error
correlation ρ, the variance-optimal weight on the first is:

```
w* = (s₂² - ρ·s₁·s₂) / (s₁² + s₂² - 2ρ·s₁·s₂)
```

With physics at 5.94, tree at 16.37, and ρ = 0.070, this gives w* = 0.887, close to the 0.87
we ship. The formula also predicted our measured gain accurately, forecasting +0.30 against a
measured +0.28, which is why we trusted it enough to plan with.

The formula carries one decisive implication: the gain depends on ρ at least as much as on
accuracy. A tree at RMSE 8 with ρ = 0.07 is worth considerably more than one at RMSE 6 with
ρ = 0.6, because the second fails on the same rows the physics fails on. We tested 18
combinations of model and feature set looking for a better member, and the best of them
improved the projected blend by **+0.021** against our 0.3 adoption bar. The tree at 16.37 is
already close to optimal *as an ensemble member*, which is a different claim from being a
good model.

### 2.3 Why brute force loses here

The problem statement says so directly: *"Brute-forcing mathematical algorithms without
understanding the underlying chemical system will not win this hackathon."* Three concrete
reasons support this, independent of the rubric.

- **Sample size.** At n = 150, hyperparameter searches mostly fit noise. We measured this: a
  properly [nested](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
  search over ExtraTrees hyperparameters improved the result by **+0.0119**, meaning the
  unsearched defaults were already as good.
- **Extrapolation.** A tree cannot predict outside its training range, whereas an ODE
  integrates whatever conditions it is handed.
- **Nothing to say afterwards.** A tuned black box cannot report the crossover temperature,
  cannot be checked against known kinetics, and cannot identify which part of the plant
  changed when it begins to fail. Section 4.5 develops this last point.

---

## 3. RMSE, understood against this experiment

### 3.1 What the metric is and why it applies here

[RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) is the square root of the
mean squared error. Because the target is a percentage, **RMSE is measured in
yield-percentage points**. An RMSE of 6 means a typical miss of about 6 yield points: if the
true value is 40%, predictions tend to land somewhere between 34% and 46%.

Squared error is preferred over [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
because squaring makes large errors disproportionately expensive.

| Size of miss | Contribution to squared error |
|---|---|
| 1 point | 1 |
| 10 points | 100 |
| 50 points | **2500** |

One row wrong by 50 costs as much as 2500 rows wrong by 1. For a chemical plant this is
arguably the correct preference, since being badly wrong once about whether the reactor is
producing anything matters more than being slightly wrong everywhere.

### 3.2 Why cliff rows dominate the score

Combining the previous section with Section 1.4 explains where our score comes from. The
target is 24.7% exact zeros and 12% above 90, with a sharp transition between them. Getting
the *side of the cliff* wrong produces an error of 40 to 90 points, which under squaring
overwhelms everything else.

We measured this on our own out-of-fold predictions. Grouping rows by how much the training
data disagrees locally, **29% of rows carry 91% of the squared error**. The well-determined
majority sits at an RMSE of 0.652, while the cliff rows sit at 10.410.

The strategic consequence is that further modelling of the well-determined majority cannot
move the score. We used this finding to decide what was worth working on.

### 3.3 Our leaderboard score, decomposed

We scored **RMSE 11.0442, MAE 3.2408, R² 0.9061**, placing 13th of 400. The two error metrics
disagree sharply, and that disagreement is itself informative.

The ratio **RMSE / MAE = 3.41** is the key figure. For Gaussian errors that ratio is 1.25,
and for the heavier-tailed [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution)
distribution it is 1.41. A ratio of 3.41 cannot arise from uniformly mediocre prediction. It
is the signature of a few catastrophic rows sitting on top of an otherwise accurate fit.
Solving for an error distribution that matches RMSE, MAE, and R² simultaneously across 50
rows gives:

| Rows missed badly | Their error | Error on the remaining rows |
|---|---|---|
| 2 | 55.0 | **1.09** |
| 3 | 45.0 | **0.57** |

In other words, roughly 47 of 50 rows were predicted to about 1 yield-point, and two or three
cliff rows were missed by around 50. The R² of 0.9061 further implies that the test targets
have a spread of 36.0 against 38.3 in training, so the test set is distributed much like the
training set.

### 3.4 How one submission with no leaderboard changes the method

A public leaderboard allows a submit-observe-adjust loop. We had none, so every choice had to
be made against internal validation. This creates a specific danger: trying 50 variants and
keeping whichever scores best on cross-validation means that score is partly luck, and the
validation set has effectively been fitted.

We defended against this in three ways.

- **[Repeated k-fold](https://scikit-learn.org/stable/modules/cross_validation.html) across
  five or more seeds**, never a single split. At n = 150, a single split moves by several
  RMSE points depending on the seed alone.
- **Pre-registered adoption bars.** We fixed the rule "+0.3 RMSE improvement, holding on
  every held-out seed" *before* running experiments, so that results could not be argued into
  significance afterwards.
- **Held-out seeds** for choosing the prediction policy itself.

One limitation surfaced late and is worth stating plainly. Leave-one-seed-out does not price
hyperparameter selection, because all seeds re-partition the same 150 rows. Measured at row
level, selecting the blend weight and cutoff costs **+0.371 RMSE**, so our figure of 5.671
should be read as a lower bound. Section 6.3 returns to this.

---

## 4. Our solution, top to bottom

### 4.1 The model

Three coupled [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation),
integrated along the reactor from inlet to outlet:

```
dCA/dz = -k1·CA                                  how fast A disappears
dCB/dz =  k1·CA - k2·CB                          B made, minus B destroyed
dT/dz  =  a1·k1·CA + a2·k2·CB + U·(T_jacket - T) reaction heat plus heat through the wall
```

The middle line is the trade-off from Section 1.2 written as an equation: B gains from the
first reaction and loses to the second, simultaneously.

The third line couples everything together. Temperature sets the rate constants through the
Arrhenius relation, the rate constants determine how much heat is released, and the jacket
term drags the temperature toward `T_jacket` at a rate governed by U. We integrate the system
numerically for each row, then take yield = 100 × CB(exit) / CA(inlet).

### 4.2 The seven parameters and their physical meaning

| Parameter | Value | Units | Physical meaning |
|---|---|---|---|
| `ln_k1_ref` | 2.7187 | none | log rate constant of A to B at the 430 K reference |
| `E1_kJ` | **43.16** | kJ/mol | activation energy of the desired reaction |
| `ln_k2_ref` | 0.1594 | none | log rate constant of B to C at 430 K |
| `E2_kJ` | **250.07** | kJ/mol | activation energy of the waste reaction; 95% CI [233.7, 269.9] |
| `a1` | **-11.79** | K·L/mol | temperature change per unit of A converted; 95% CI [-12.12, -11.49] |
| `a2` | **+11.34** | K·L/mol | temperature change per unit of B destroyed |
| `U` | **3.2552** | per unit length | how strongly the jacket pulls temperature toward itself |

The signs of a₁ and a₂ give the direction of heat flow, with a negative value cooling the
fluid and a positive value heating it. The confidence intervals come from
[profile likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Profile_likelihood)
with an F-test criterion: pin one parameter across a grid, refit all the others, and measure
how much the fit degrades.

One qualification appears in all our documents. An eighth parameter, `n_flow`, a
flow-dependence exponent on U, floats during cross-validation but is pinned to zero in the
shipped model. Our cross-validation figures therefore describe eight free parameters for a
seven-parameter model. Its 95% interval of [-0.032, 0.016] is indistinguishable from zero, so
it changes no conclusion, but the discrepancy is stated rather than hidden.

> If asked why the rate constants are referenced to 430 K, the answer is that fitting `ln A`
> and `E` directly makes them correlate above 0.999, which turns the fitting surface into a
> long narrow valley that optimisers crawl along. Centering on 430 K decorrelates them.

### 4.3 Noise-averaging at σ = 1.67 K

Rather than predicting `f(inputs)` directly, we predict the average of `f` over a small
spread of possible inputs, specifically a Gaussian of width σ = 1.67 K applied to both
temperatures.

The reasoning is as follows. If the recorded temperatures carry small measurement errors,
then the honest prediction is not the yield at exactly 430.0 K but the expected yield given
that the true temperature lies somewhere near 430 K. Where the response is curved, which is
precisely at the cliff, these two quantities differ substantially.

The origin of the value 1.67 matters. It comes from an
[errors-in-variables](https://en.wikipedia.org/wiki/Errors-in-variables_model) fit, in which
we solved for the temperature offset that would make our model reproduce each observation
exactly. **138 of 150 rows** are explainable this way, with a median offset of **1.67 K**,
and those offsets show no structure (the largest feature correlation is 0.206). We fixed σ at
that value *before* cross-validating it, so it was not tuned on the score that judges it.

The evidence that the effect is real comes from an asymmetry. Noise-averaging makes the
*training* fit worse, moving it from 3.656 to 3.764, while making cross-validation better,
moving it from 6.358 to 5.928 on all three seeds. Blurring a correct model with clean inputs
would damage both. Improving held-out error while degrading training error is the opposite of
what overfitting looks like.

> The 12 rows that resist any offset within ±25 K remain unexplained. Say so if asked. We do
> not claim the residual is fully accounted for.

### 4.4 The tree blend and what it costs

We blend an [ExtraTrees](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees)
model at 13% weight, but only on rows where the physics predicts below 60% yield. Above that
threshold we use pure physics, because trees cannot extrapolate past their outermost split
and can therefore only drag high-yield predictions toward the training mean.

What the tree provides is variance reduction through averaging, not bias correction. It is
much worse on its own, scoring 15.04 against 6.14 for the physics on the same rows, but its
errors are nearly uncorrelated with the physics errors, with ρ = 0.070. Averaging two
estimators whose mistakes are independent reduces variance even when one of them is
substantially weaker.

What we could not show is that the tree corrects a systematic bias. A
[sign test](https://en.wikipedia.org/wiki/Sign_test) on the band carrying the gain returned
**8 of 13** aligned, with p = 0.581, against a pre-registered bar of 11 of 13. This is a
failure to demonstrate rather than proof of absence, since at n = 13 the test detects only
very large effects, so we claim the weaker interpretation.

The cost is real and worth stating precisely. On the near-zero rows the blend makes matters
worse, with slice RMSE rising from 0.366 to 1.960. It lifts six dead-reactor predictions
above 5% yield, with a maximum of 11.499. We keep it because the net effect across all
held-out seeds is +0.27 and the weight sits on a plateau rather than a peak, but the cost is
not negligible.

> If asked why there is a random forest inside a physics model: it is a weak, decorrelated
> component that reduces ensemble variance by averaging, its weight was derived from the
> error correlation rather than tuned, and we tested the principled alternative of bagging
> the physics fit, which recovered only +0.13 and destabilised one seed.

### 4.5 What the mechanistic model provides beyond accuracy

Because the parameters are physical, a failure becomes diagnosable. Refitting the seven
parameters on new data and observing which one moved identifies the cause.

| What moves | Diagnosis |
|---|---|
| U alone | heat-transfer fouling |
| k₁ alone | catalyst degradation |
| Only the concentration path | feed composition change |
| Nothing moves, but errors rise | sensor drift or unmodelled physics |

A black-box model can tell you that the prediction is wrong. It cannot tell you which piece
of the plant changed.

---

## 5. What we did

### 5.1 The sequence of work

1. **Established a baseline.** Tree ensembles on raw features scored 19.50, and on
   physics-derived features 16.37. This is the figure the physics had to beat.
2. **Wrote the ODEs and built an integrator.** Fixed-step
   [RK4](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) diverged on
   thermal-runaway rows, so we moved to an operator-splitting scheme that advances the mass
   balance analytically and the heat balance with its exact linear solution. Both halves are
   unconditionally stable.
3. **Fitted the seven parameters** by [global optimisation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
   followed by local polish, reaching train RMSE 3.66 and 10-fold cross-validation 6.36.
4. **Tested alternative mechanisms**, all of which were rejected on measurement (Section 5.2).
5. **Investigated the residual**, found the input-noise evidence, and adopted σ = 1.67 K.
6. **Derived the blend policy** for σ, weight, and cutoff jointly rather than one at a time.
7. **Submitted**, then continued testing eight further improvement strategies, none of which
   were adopted.

### 5.2 Where we failed, and what each failure taught

| What we tried | Result | What it taught |
|---|---|---|
| Fixed-step RK4 integrator | Diverged on hot rows | Stability matters more than order for stiff problems |
| Early fit with a loose convergence guard | Parameters moved 92 yield-points when the step count changed | The optimiser was fitting integration error rather than data. Gate the numerics before trusting any fit |
| First cross-validation | Reported 3.662 ± **0.000** | A zero seed-to-seed spread is a warning sign, not a triumph; the folds could not move |
| Parallel A-to-C path | Train +0.0019, CV **-0.352** | Negligible train gain with degraded generalisation is the signature of fitting noise |
| Residual ML correction | +0.027 against a 0.3 bar | Our residual carries little learnable structure |
| Bagging the physics fit | +0.13 of +0.28, one seed worse | Resampling on this dataset lands in a second parameter basin |
| Better features for the tree | Best +0.021 against a 0.3 bar | The tree at 16.37 is already near-optimal as an ensemble member |
| Two-stage hurdle model | LOSO -0.076 | Tested after submission, too late to matter either way |

The second row is the most important lesson of the project. An optimiser will exploit
numerical error if permitted to, and the result looks like a good fit.

### 5.3 Where we spent extra effort, and why

- **Numerical trust.** We ran an independent SciPy BDF check on every integrator change.
  Without it we could not distinguish a wrong model from a wrong solver.
- **Identifiability.** We computed profile likelihoods for the parameters that mattered,
  since a fitted value means little without knowing how tightly the data pins it down.
- **Honest validation.** We used cold-start folds refit from scratch, because warm-started
  cross-validation flatters itself.
- **Falsifying our own claims.** Several documented results are corrections of things we had
  previously asserted and later disproved.

### 5.4 Our diagnostics and the question each one answers

| Script | Question it answers | Result |
|---|---|---|
| `check_integrator.py` | Is the solver actually solving the equations? | Agrees with SciPy BDF to 0.02 yield-points |
| `diagnose_fit.py` | Is a bad fit the optimiser's fault or the model's? | Recovers known parameters to 5 decimals, so the optimiser is sound |
| `check_cold_folds.py` | Is our cross-validation optimistic? | Held-out 2.97, 13.05, 1.93, mean 5.98 |
| `profile_params.py` | How well is each parameter determined? | E₂ 95% CI [233.7, 269.9] |
| `check_bounds.py` | Are any values sitting at their constraints? | A widened box returns the same optimum |
| `eiv_noise.py` | Are the inputs noisy? | 138 of 150 rows explained, median 1.67 K |
| `noise_averaged.py` | Does smoothing improve held-out error? | 6.358 to 5.928, on all three seeds |
| `tree_sign_test.py` | Bias correction or variance reduction? | 8 of 13, p = 0.581, so variance reduction |
| `blend_accounting.py` | Does the reported gain reconcile? | Band contributions sum to the total, residual 4.6e-13 |
| `audit_pitch.py` | Do our slides match our artifacts? | Exit code 0, or the build fails |

> The self-recovery test in `diagnose_fit.py` is the one to name if asked how we knew our
> fits were trustworthy. It regenerates the targets from known parameters and refits from
> scratch. If it cannot recover them, nothing downstream carries any meaning.

---

## 6. Where we stand

### 6.1 Our standing after Round 1

**Rank 13 of 400**, with RMSE 11.0442, MAE 3.2408, and R² 0.9061. As decomposed in Section
3.3, this corresponds to roughly 47 rows accurate to about 1 yield-point and two or three
cliff rows missed by around 50.

The offline round carries 200 marks, weighted as Chemical Engineering 60, ML methodology 45,
leaderboard 40, validation and failure analysis 30, and presentation and Q&A 25. The
leaderboard mark is already banked. The remaining 160 marks concern understanding, which is
what this document exists to support.

### 6.2 Our strongest points

1. **The ratio E₂/E₁ = 5.79 and the 449.9 K crossover**, computed from the fitted parameters
   and lying inside the observed envelope. The correct qualification is that this is a joint
   condition on temperature and residence time rather than a temperature ceiling: 49 training
   rows sit above the crossover and one still reaches 75% yield, because at short residence
   time B leaves the reactor before it degrades.
2. **The ODE out-classifies every classifier at distinguishing dead from alive reactors.**
   Pooled AUC gives physics **0.9587**, ahead of ExtraTrees at 0.9505, logistic regression at
   0.9253, and gradient boosting at 0.9138. Seven physical parameters beat every
   discriminative model at the one task classifiers are supposed to own. *(Internal caveat:
   these AUCs come from the post-submission sweep and are the only figures in this document
   without a stored JSON artifact. We can regenerate them, but cannot point at a file.)*
3. **Independent blind replication.** A separate fit using a different integrator and
   optimiser, with no access to our code, produced E₁ = 43.156 against our 43.161, E₂ = 250.53
   against our 250.07, and train RMSE 3.6554 against our 3.6559.
4. **The failure diagnostic** of Section 4.5, which is the practical argument for a
   mechanistic model over an approximation.
5. **We falsified our own claims** repeatedly, and documented each correction.

### 6.3 Where we fell short

**Our RMSE fell outside our own predicted band.** Bootstrapping 50-row draws from our
out-of-fold predictions gave a range of [2.94, 8.10], and we scored 11.04. Under our own
model that outcome carried a probability of **0.0003**, and our out-of-fold predictions
contained **no rows at all** with an error above 40, whereas the test set produced two or
three. We identified the failure *mechanism* correctly and underestimated its *magnitude*.

> If pressed on this, concede it directly and do not defend the band. The supporting evidence
> is that our MAE prediction was accurate, with an observed 3.24 against a predicted range of
> [1.74, 4.03], so the model was sound on typical rows and wrong about the tail. That is the
> honest limit of estimating a tail from 150 training rows.

**Our headline validation number is optimistic.** Leave-one-seed-out underprices
hyperparameter selection by roughly a factor of 15, with a row-level split-half giving +0.371
against LOSO's +0.02. The policy itself survives, since every search lands at a weight between
0.85 and 0.89 with a cutoff of 60, but 5.671 is a lower bound rather than an unbiased
estimate.

**The blend predicts non-zero yield on dead reactors**, with six rows above 5% and a maximum
of 11.499.

**We tested the hurdle model too late.** A two-stage dead-or-alive classifier followed by a
regressor is the natural approach to a zero-inflated target, and we tried it only after
submitting. It failed, at LOSO -0.076, so nothing was lost, but we did not know that at the
time it would have mattered.

**Given more time**, the highest-value work would be sampling more of the cliff region. Our
entire error budget lives where we have very few rows, and 10 to 20 additional rows there
would be worth more than any modelling change we tested.

---

## Traceability

Every number in this document comes from a stored artifact.

| Figure | Source |
|---|---|
| Seven fitted parameters, train RMSE 3.6559 and 3.6617 | `artifacts/physics_params.json` |
| Policy: σ 1.67, weight 0.87, cutoff 60 | `artifacts/blend.json` |
| LOSO 5.671 against 6.022 | `artifacts/joint_policy.json` |
| Tree baselines 16.37 and 19.50 | `artifacts/baseline_cv.json` |
| Variant comparison, neutral 8.27 | `artifacts/model_comparison.json` |
| Cold folds 2.97, 13.05, 1.93 | `artifacts/cold_folds.json` |
| Crossover 449.9 K, E₂/E₁ 5.79, timings, AUC, parallel path | `artifacts/pitch_evidence.json` |
| F-test intervals | `artifacts/pitch_evidence.json`, key `ftest` |
| Sign test 8 of 13, band table | `artifacts/tree_sign_test.json` |
| Band SSE reconciliation | `artifacts/blend_accounting.json` |
| EIV 138 of 150, median 1.67 K | `artifacts/eiv_offsets.json` |
| Noise-averaging cross-validation by σ | `artifacts/noise_averaged.json` |
| Leaderboard decomposition and bootstrap | `artifacts/score_decomposition.json` |
| Feature and learner sweeps | `artifacts/feature_eval.json`, `learner_eval.json` |
| Operating rule: 49 hot rows, 75% yield | `artifacts/operating_rule.json` |
| Residual corrector rejection | `artifacts/residual_verification.json` |

**Further reading in this repository.** `CLAUDE.md` holds the full engineering record,
including every rejected hypothesis. `docs/pitch/appendix.md` has the complete parameter and
protocol tables. `docs/pitch/qa_bank.md` holds 48 rehearsed Q&A answers.
`docs/pitch/deck.md` is the 12-slide backup deck for the Q&A round, and
`docs/pitch/slides.md` is what actually gets presented.
