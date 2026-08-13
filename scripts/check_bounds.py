"""Is the optimum pinned against a box constraint?

E1 lands at ~43 kJ/mol against a lower bound of 40, and E2 at ~250 against an
upper bound of 280. A parameter resting on its bound is not a recovered physical
quantity -- it is the optimizer being cut off. This refits with the box widened
well past both, and reports how far each parameter sits from its constraint.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_train, ode_inputs, rmse
from src.physics import LOWER, PARAM_NAMES, SEARCH_STEPS, UPPER, DEFAULT_STEPS, integrate, objective, residuals

ARTIFACTS = ROOT / "artifacts"

# n_flow stays pinned at 0 (the comparison put it at -0.03, i.e. inactive).
WIDE_LOWER = np.array([-25.0, 5.0, -25.0, 20.0, -60.0, -60.0, 0.0, 0.0])
WIDE_UPPER = np.array([25.0, 400.0, 25.0, 600.0, 60.0, 60.0, 200.0, 0.0])


def main() -> int:
    df = load_train()
    inputs = ode_inputs(df)
    y = df[TARGET].to_numpy(dtype=float)

    free = [i for i, n in enumerate(PARAM_NAMES) if n != "n_flow"]
    lo, hi = WIDE_LOWER[free], WIDE_UPPER[free]

    def expand(xf):
        full = np.zeros(len(PARAM_NAMES))
        full[free] = xf
        return full

    def obj(xf):
        return objective(expand(xf), inputs, y, SEARCH_STEPS, 0.05)

    def res_fn(xf):
        return residuals(expand(xf), inputs, y, DEFAULT_STEPS, None)

    t0 = time.perf_counter()
    de = differential_evolution(
        obj, bounds=list(zip(lo, hi)), maxiter=300, popsize=20, tol=1e-9,
        mutation=(0.3, 1.2), recombination=0.85, seed=0,
        polish=False, init="sobol", updating="deferred",
    )
    best_x, best = de.x, float(de.fun)
    rng = np.random.default_rng(0)
    for k in range(6):
        x0 = de.x if k == 0 else np.clip(de.x + rng.normal(0, 0.03, de.x.shape) * (hi - lo), lo, hi)
        try:
            r = least_squares(res_fn, x0=x0, bounds=(lo, hi), x_scale="jac",
                              xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=800)
        except Exception:
            continue
        s = rmse(y, integrate(expand(r.x), inputs, n_steps=DEFAULT_STEPS))
        if s < best:
            best, best_x = s, r.x

    print(f"wide-bounds refit: train RMSE {best:.4f}   ({time.perf_counter() - t0:.0f}s)")
    print("(narrow-bounds result was 3.6617)\n")
    print(f"{'param':>10s} {'value':>12s} {'wide lo':>10s} {'wide hi':>10s} {'narrow lo':>10s} {'narrow hi':>10s}  at bound?")
    for k, i in enumerate(free):
        v = best_x[k]
        span = hi[k] - lo[k]
        pinned = (v - lo[k]) < 0.01 * span or (hi[k] - v) < 0.01 * span
        print(f"{PARAM_NAMES[i]:>10s} {v:12.4f} {lo[k]:10.1f} {hi[k]:10.1f} "
              f"{LOWER[i]:10.1f} {UPPER[i]:10.1f}  {'YES' if pinned else 'no'}")

    (ARTIFACTS / "wide_bounds.json").write_text(json.dumps({
        "train_rmse": best,
        "vector": [float(v) for v in expand(best_x)],
        "params": dict(zip(PARAM_NAMES, (float(v) for v in expand(best_x)))),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
