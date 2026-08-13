"""Is the reported CV number real, or an artifact of warm-starting?

`make_physics_predict_fn` starts each fold's least_squares from the *full-data*
optimum. On 135 of 150 rows that start is already near-optimal, so the "refit"
barely moves and OOF RMSE collapses onto train RMSE -- which is exactly what we
saw (3.662 vs 3.6617, std 0.000 across seeds). A std of exactly zero is not
evidence of a stable model; it is evidence the folds were not independent of the
rows they were scored on.

This refits **cold** (differential evolution from scratch, no warm start) on
held-out subsets and compares the recovered parameters and held-out error to the
full-data fit. Cold folds are genuinely independent, so their numbers are the
honest ones.

diagnose_fit.py already showed DE recovers the global optimum reliably, so a
handful of folds is informative -- this does not need all ten.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_train, ode_inputs, rmse
from src.physics import (
    DEFAULT_STEPS,
    LOWER,
    PARAM_NAMES,
    SEARCH_STEPS,
    UPPER,
    integrate,
    objective,
    residuals,
)

ARTIFACTS = ROOT / "artifacts"
FREE = [i for i, n in enumerate(PARAM_NAMES) if n != "n_flow"]


def cold_fit(inputs, y, seed, maxiter, popsize):
    lo, hi = LOWER[FREE], UPPER[FREE]

    def expand(xf):
        full = np.zeros(len(PARAM_NAMES))
        full[FREE] = xf
        return full

    de = differential_evolution(
        lambda xf: objective(expand(xf), inputs, y, SEARCH_STEPS, 0.05),
        bounds=list(zip(lo, hi)), maxiter=maxiter, popsize=popsize, tol=1e-9,
        mutation=(0.3, 1.2), recombination=0.85, seed=seed,
        polish=False, init="sobol", updating="deferred",
    )
    best_x, best = de.x, float(de.fun)
    for k in range(3):
        rng = np.random.default_rng(seed + k)
        x0 = de.x if k == 0 else np.clip(de.x + rng.normal(0, 0.03, de.x.shape) * (hi - lo), lo, hi)
        try:
            r = least_squares(lambda xf: residuals(expand(xf), inputs, y, DEFAULT_STEPS, None),
                              x0=x0, bounds=(lo, hi), x_scale="jac",
                              xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=800)
        except Exception:
            continue
        s = rmse(y, integrate(expand(r.x), inputs, n_steps=DEFAULT_STEPS))
        if s < best:
            best, best_x = s, r.x
    return expand(best_x), best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3, help="how many of the 10 folds to run cold")
    ap.add_argument("--maxiter", type=int, default=250)
    ap.add_argument("--popsize", type=int, default=16)
    args = ap.parse_args()

    df = load_train()
    y = df[TARGET].to_numpy(dtype=float)
    x_full = np.array(json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"])

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    rows, held_errs = [], []
    for i, (tr_idx, te_idx) in enumerate(kf.split(df)):
        if i >= args.folds:
            break
        t0 = time.perf_counter()
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        x_fold, fold_train_rmse = cold_fit(ode_inputs(tr), tr[TARGET].to_numpy(dtype=float),
                                           seed=i, maxiter=args.maxiter, popsize=args.popsize)
        held = rmse(te[TARGET].to_numpy(dtype=float),
                    integrate(x_fold, ode_inputs(te), n_steps=DEFAULT_STEPS))
        held_errs.append(held)
        rows.append(x_fold)
        print(f"fold {i}: cold refit on {len(tr)} rows -> train {fold_train_rmse:.4f}, "
              f"HELD-OUT {held:.4f}   ({time.perf_counter() - t0:.0f}s)", flush=True)

    M = np.array(rows)
    print(f"\nheld-out RMSE across {len(held_errs)} cold folds: "
          f"mean {np.mean(held_errs):.4f}, max {np.max(held_errs):.4f}")
    print("(full-data train RMSE was 3.6617; warm-started CV reported 3.662 +/- 0.000)\n")

    print(f"{'param':>10s} {'full data':>12s} {'cold folds mean':>16s} {'max % shift':>12s}")
    shifts = {}
    for k, i in enumerate(FREE):
        col = M[:, i]
        denom = abs(x_full[i]) if abs(x_full[i]) > 1e-9 else 1.0
        shift = 100 * np.max(np.abs(col - x_full[i])) / denom
        shifts[PARAM_NAMES[i]] = float(shift)
        print(f"{PARAM_NAMES[i]:>10s} {x_full[i]:12.4f} {col.mean():16.4f} {shift:11.2f}%")

    worst = max(shifts.values())
    print()
    if worst < 5.0:
        print(f"VERDICT: dropping 10% of the data moves every parameter by <{worst:.1f}%.")
        print("         The stability claim is REAL and can be quoted directly.")
    else:
        print(f"VERDICT: parameters move up to {worst:.1f}% -- the warm-started 3.662")
        print("         is optimistic. Quote the cold held-out number instead.")

    (ARTIFACTS / "cold_folds.json").write_text(json.dumps({
        "held_out_rmse": [float(v) for v in held_errs],
        "held_out_mean": float(np.mean(held_errs)),
        "param_max_pct_shift": shifts,
        "fold_vectors": [[float(v) for v in r] for r in rows],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
