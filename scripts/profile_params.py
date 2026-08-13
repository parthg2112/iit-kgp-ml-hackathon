"""Profile-likelihood for the weakly-identified parameters.

An external audit's independent fit put E2 near 155 kJ/mol; ours lands at 250. Our fit is
far better on the same data (train RMSE 3.66 vs their 11.3), but E2 is also our *least*
identified parameter -- it shifted 11.6% across cold folds, and ln_k2/a2 shifted 61%/91%.
So "our fit is better" is not by itself an answer to "is 155 admissible?".

This pins a parameter at each point of a grid, refits everything else, and records the
resulting train RMSE. That curve is the honest statement of how well the data determines
it. Also tracked: the prediction for test row 0, whose value depends on the local
concentration sensitivity that a1 and E2 jointly control, and which is the single largest
disagreement with the audit.

Usage:  python scripts/profile_params.py --param E2_kJ
        python scripts/profile_params.py --param a1
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_test, load_train, ode_inputs, rmse
from src.physics import (
    CONVERGENCE_TOL,
    DEFAULT_STEPS,
    LOWER,
    PARAM_NAMES,
    SEARCH_STEPS,
    UPPER,
    integrate,
    pin_expand,
    pin_free_indices,
    pinned_objective,
    pinned_residuals,
)

ARTIFACTS = ROOT / "artifacts"
GRIDS = {
    "E2_kJ": [100, 130, 155, 180, 210, 235, 250, 265, 290, 320],
    "a1": [-25, -20, -15, -11.79, -8, -5, -2, 0, 5],
    "E1_kJ": [25, 35, 43.16, 55, 70, 90],
    # Turbulent internal flow gives h ~ Re^0.8, so a Reynolds-driven wall
    # coefficient predicts n_flow ~ 0.8. The unconstrained fit returns -0.03;
    # profiling turns "our fit found nothing" into a stated interval, which is
    # evidence about the reactor rather than a failed search.
    "n_flow": [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}


def profile_point(name, value, inputs, y, maxiter, popsize, seed=0):
    """Refit every other parameter with `name` pinned at `value`.

    Runs single-threaded on purpose: the grid points are independent, so the
    parallelism lives one level up (a process per point) rather than inside each
    differential_evolution call. That keeps all cores busy without oversubscribing.
    """
    # n_flow is normally held at 0 (the plain plug-flow model). When n_flow is
    # itself the parameter being profiled it must be pinned once, at the grid
    # value -- pinning it twice would silently overwrite the grid value with 0.
    if name == "n_flow":
        pin_names, pin_values = ("n_flow",), (float(value),)
    else:
        pin_names, pin_values = (name, "n_flow"), (float(value), 0.0)
    free = pin_free_indices(pin_names)
    lo, hi = LOWER[free], UPPER[free]

    de = differential_evolution(
        pinned_objective,
        bounds=list(zip(lo, hi)),
        args=(pin_names, pin_values, inputs, y, SEARCH_STEPS, CONVERGENCE_TOL),
        maxiter=maxiter, popsize=popsize, tol=1e-9,
        mutation=(0.3, 1.2), recombination=0.85, seed=seed,
        polish=False, init="sobol", updating="deferred",
    )
    best_x, best = de.x, float(de.fun)
    rng = np.random.default_rng(seed)
    for k in range(3):
        x0 = de.x if k == 0 else np.clip(de.x + rng.normal(0, 0.03, de.x.shape) * (hi - lo), lo, hi)
        try:
            r = least_squares(pinned_residuals, x0=x0, bounds=(lo, hi),
                              args=(pin_names, pin_values, inputs, y, DEFAULT_STEPS, None),
                              x_scale="jac", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=700)
        except Exception:
            continue
        s = rmse(y, integrate(pin_expand(r.x, pin_names, pin_values), inputs, n_steps=DEFAULT_STEPS))
        if s < best:
            best, best_x = s, r.x
    return pin_expand(best_x, pin_names, pin_values), best


def _worker(job):
    """Top-level so multiprocessing can pickle it."""
    name, value, inputs, y, maxiter, popsize = job
    t0 = time.perf_counter()
    x, s = profile_point(name, value, inputs, y, maxiter, popsize)
    return value, x, s, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default="E2_kJ", choices=sorted(GRIDS))
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--popsize", type=int, default=14)
    args = ap.parse_args()

    tr, te = load_train(), load_test()
    inputs, te_inputs = ode_inputs(tr), ode_inputs(te)
    y = tr[TARGET].to_numpy(dtype=float)
    ref = json.loads((ARTIFACTS / "physics_params.json").read_text())
    ref_rmse = ref["train_rmse"]

    print(f"profiling {args.param}; unconstrained optimum = "
          f"{ref['params'][args.param]:.2f} at train RMSE {ref_rmse:.4f}\n")
    grid = GRIDS[args.param]
    n_proc = min(len(grid), max(1, cpu_count() - 1))
    print(f"running {len(grid)} points across {n_proc} processes "
          f"({cpu_count()} cores available)\n")
    print(f"{args.param:>10s} {'train RMSE':>11s} {'excess':>8s} {'row0':>8s} {'row24':>8s} {'row39':>8s}")

    jobs = [(args.param, v, inputs, y, args.maxiter, args.popsize) for v in grid]
    out = {}
    t_all = time.perf_counter()
    with Pool(n_proc) as pool:
        for v, x, s, secs in pool.imap_unordered(_worker, jobs):
            pred = np.clip(integrate(x, te_inputs, n_steps=DEFAULT_STEPS), 0, 100)
            out[str(v)] = {"train_rmse": s, "vector": [float(t) for t in x],
                           "test_pred": [float(t) for t in pred]}
            print(f"{v:10.2f} {s:11.4f} {s - ref_rmse:+8.4f} "
                  f"{pred[0]:8.2f} {pred[24]:8.2f} {pred[39]:8.2f}   ({secs:.0f}s)", flush=True)
    print(f"\nwall clock: {time.perf_counter() - t_all:.0f}s")

    # Admissible region: an F-test-flavoured threshold. With n=150 and ~7 parameters, a
    # ~10% inflation in RMSE is a generous bar for "this value is still consistent".
    thresh = ref_rmse * 1.10
    ok = [float(v) for v in GRIDS[args.param] if out[str(v)]["train_rmse"] <= thresh]
    print(f"\nadmissible at RMSE <= {thresh:.3f} (10% above optimum): {ok}")
    if ok:
        print(f"  range: {min(ok)} .. {max(ok)}")
    rows = [(v, out[str(v)]["test_pred"][0]) for v in GRIDS[args.param]
            if out[str(v)]["train_rmse"] <= thresh]
    if rows:
        vals = [r[1] for r in rows]
        print(f"  test row 0 across the admissible region: {min(vals):.1f} .. {max(vals):.1f}")

    (ARTIFACTS / f"profile_{args.param}.json").write_text(json.dumps(
        {"param": args.param, "reference_rmse": ref_rmse, "points": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
