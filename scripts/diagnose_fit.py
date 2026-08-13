"""Two diagnostics that decide whether to add mechanism or fix the search.

1. SELF-RECOVERY. Generate synthetic targets from the fitted parameters on the
   real 150 X rows, then refit from scratch with the same budget. If the fit
   cannot recover parameters it is *known* to have generated, the search is the
   bottleneck and no extra mechanism will help. If it recovers RMSE ~ 0, then
   the 4.38 on real data is genuine model-form error.

2. POISON-HIT COUNT. `residuals()` returns a flat 1e3 vector when the step
   convergence check trips. least_squares builds its Jacobian by finite
   differences, so a poison hit anywhere near the optimum produces a garbage
   derivative and a meaningless step -- a candidate explanation for 18 of 24
   polish starts landing at RMSE 12-30.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_train, ode_inputs, rmse
from src.physics import (
    CONVERGENCE_TOL,
    DEFAULT_STEPS,
    LOWER,
    PARAM_NAMES,
    POISON,
    SEARCH_STEPS,
    UPPER,
    integrate,
    objective,
    residuals,
)

ARTIFACTS = ROOT / "artifacts"


def poison_probe(x_true, inputs, y):
    """How often does the convergence guard fire around the optimum?"""
    hits = 0
    trials = 0
    span = UPPER - LOWER
    rng = np.random.default_rng(0)

    # Finite-difference-sized perturbations, i.e. exactly what least_squares probes.
    for eps in (1e-8, 1e-6, 1e-4, 1e-2):
        for j in range(len(x_true)):
            for sign in (+1, -1):
                xp = np.array(x_true, dtype=float)
                xp[j] = np.clip(xp[j] + sign * eps * max(1.0, abs(xp[j])), LOWER[j], UPPER[j])
                r = residuals(xp, inputs, y, n_steps=DEFAULT_STEPS)
                trials += 1
                hits += int(np.allclose(r, POISON))

    # And a wider random cloud, the region a polish start wanders through.
    for _ in range(200):
        xp = np.clip(np.array(x_true) + rng.normal(0, 0.04, len(x_true)) * span, LOWER, UPPER)
        r = residuals(xp, inputs, y, n_steps=DEFAULT_STEPS)
        trials += 1
        hits += int(np.allclose(r, POISON))

    return hits, trials


def main() -> int:
    df = load_train()
    inputs = ode_inputs(df)
    y_real = df[TARGET].to_numpy(dtype=float)
    x_true = np.array(json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"])

    print("=" * 64)
    print("DIAGNOSTIC 2: poison-hit count around the fitted optimum")
    print("=" * 64)
    hits, trials = poison_probe(x_true, inputs, y_real)
    print(f"convergence guard fired {hits}/{trials} times (tol={CONVERGENCE_TOL})")
    if hits:
        print("  -> the guard IS corrupting the Jacobian near the optimum.")
        print("     Disable it during polish, or replace the hard poison with a smooth penalty.")
    else:
        print("  -> guard never fires inside the bounds here; it is not the polish problem.")

    print()
    print("=" * 64)
    print("DIAGNOSTIC 1: self-recovery on synthetic data")
    print("=" * 64)
    y_syn = integrate(x_true, inputs, n_steps=DEFAULT_STEPS)
    print(f"synthetic target from fitted params: range [{y_syn.min():.2f}, {y_syn.max():.2f}]")

    t0 = time.perf_counter()
    de = differential_evolution(
        objective,
        bounds=list(zip(LOWER, UPPER)),
        args=(inputs, y_syn, SEARCH_STEPS),
        maxiter=1200, popsize=24, tol=1e-8,
        mutation=(0.3, 1.2), recombination=0.85,
        seed=0, polish=False, init="sobol",
        updating="deferred", workers=-1,
    )
    res = least_squares(
        residuals, x0=de.x, bounds=(LOWER, UPPER),
        args=(inputs, y_syn, DEFAULT_STEPS),
        x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=8000,
    )
    r_syn = rmse(y_syn, integrate(res.x, inputs, n_steps=DEFAULT_STEPS))
    print(f"DE {de.fun:.5f} -> polish {r_syn:.5f}   ({time.perf_counter() - t0:.0f}s)")

    print(f"\n{'param':>12s} {'true':>12s} {'recovered':>12s}")
    for i, name in enumerate(PARAM_NAMES):
        print(f"{name:>12s} {x_true[i]:12.5f} {res.x[i]:12.5f}")

    print()
    if r_syn < 0.5:
        print("VERDICT: search recovers a known solution. The 4.38 on real data is")
        print("         genuine MODEL-FORM error -> adding mechanism is justified.")
    else:
        print("VERDICT: search cannot recover parameters it generated itself. The")
        print("         OPTIMIZER is the bottleneck -> fix the search before adding")
        print("         any parameter.")

    (ARTIFACTS / "diagnose_fit.json").write_text(json.dumps({
        "poison_hits": hits, "poison_trials": trials,
        "self_recovery_rmse": r_syn,
        "recovered": [float(v) for v in res.x],
        "true": [float(v) for v in x_true],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
