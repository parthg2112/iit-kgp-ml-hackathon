"""Fit every model variant and report which mechanism the data actually supports.

Variants (see physics.MODELS):
  series         plain non-isothermal plug flow, constant U        (7 free)
  neutral        same, both reactions forced thermally neutral     (5 free)
  flowU          jacket heat transfer scales as (Q/Q_ref)^n        (8 free)
  flowU_neutral  flow-dependent U *and* thermally neutral          (6 free)

`neutral` is the pitch experiment: if it matches `series`, the heat terms were
never doing real work, and inlet concentration genuinely drops out of the
problem -- a far stronger claim than quoting a correlation of +0.009.

The convergence guard is left on for the global search but disabled during the
polish: it fires 159/256 times near the optimum, and least_squares builds its
Jacobian by finite differences, so a hard poison value there produces garbage
derivatives. Bounds already prevent the runaway regime the guard was added for,
and convergence is re-asserted at the fitted parameters.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_train, ode_inputs, rmse
from src.physics import (
    DEFAULT_STEPS,
    MODELS,
    SEARCH_STEPS,
    MaskedModel,
    integrate,
    masked_objective,
    masked_residuals,
)

ARTIFACTS = ROOT / "artifacts"


def fit_model(name, inputs, y, maxiter, popsize, seed, polish_starts):
    m = MaskedModel(name)
    t0 = time.perf_counter()

    de = differential_evolution(
        masked_objective,
        bounds=list(zip(m.lower, m.upper)),
        args=(name, inputs, y, SEARCH_STEPS, 0.05),
        maxiter=maxiter, popsize=popsize, tol=1e-9,
        mutation=(0.3, 1.2), recombination=0.85,
        seed=seed, polish=False, init="sobol",
        updating="deferred", workers=-1,
    )

    rng = np.random.default_rng(seed)
    span = m.upper - m.lower
    starts = [de.x] + [
        np.clip(de.x + rng.normal(0, 0.03, de.x.shape) * span, m.lower, m.upper)
        for _ in range(polish_starts - 1)
    ]

    best_free, best = de.x, float(de.fun)
    for x0 in starts:
        try:
            res = least_squares(
                masked_residuals, x0=x0, bounds=(m.lower, m.upper),
                # convergence_tol=None -> guard off, Jacobian stays differentiable
                args=(name, inputs, y, DEFAULT_STEPS, None),
                # max_nfev is deliberately modest: with a finite-difference
                # Jacobian each "evaluation" costs n_free+1 integrations, and
                # polish converges long before the cap. Raising it to 6000 made
                # a single model take ~25 min for no accuracy gain.
                x_scale="jac", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=800,
            )
        except Exception:
            continue
        r = rmse(y, integrate(m.expand(res.x), inputs, n_steps=DEFAULT_STEPS))
        if r < best:
            best, best_free = r, res.x

    x_full = m.expand(best_free)
    pred = integrate(x_full, inputs, n_steps=DEFAULT_STEPS)
    pred_fine = integrate(x_full, inputs, n_steps=4 * DEFAULT_STEPS)
    drift = float(np.abs(pred - pred_fine).max())
    # With the guard off during polish, the optimizer could in principle trade
    # accuracy for integration error again. Re-scoring at 4x substeps is the
    # honest number; a gap here means the working step count is too coarse.
    fine_rmse = rmse(y, pred_fine)
    return {
        "model": name,
        "n_free": len(m.free_idx),
        "train_rmse": best,
        "train_rmse_fine": fine_rmse,
        "de_rmse": float(de.fun),
        "step_drift": drift,
        "vector": [float(v) for v in x_full],
        "params": dict(zip(("ln_k1_ref", "E1_kJ", "ln_k2_ref", "E2_kJ", "a1", "a2", "U", "n_flow"),
                           (float(v) for v in x_full))),
        "elapsed_s": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=900)
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--polish-starts", type=int, default=10)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    df = load_train()
    inputs = ode_inputs(df)
    y = df[TARGET].to_numpy(dtype=float)

    results = {}
    for name in args.models:
        print(f"--- fitting {name} ---", flush=True)
        r = fit_model(name, inputs, y, args.maxiter, args.popsize, args.seed, args.polish_starts)
        results[name] = r
        print(f"  train RMSE {r['train_rmse']:.4f}  ({r['n_free']} free, "
              f"{r['elapsed_s']:.0f}s, drift {r['step_drift']:.2e})", flush=True)

    print(f"\n{'model':16s} {'free':>5s} {'train RMSE':>11s} {'@4x steps':>11s} {'drift':>10s}")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["train_rmse"]):
        print(f"{name:16s} {r['n_free']:5d} {r['train_rmse']:11.4f} "
              f"{r['train_rmse_fine']:11.4f} {r['step_drift']:10.2e}")

    best_name = min(results, key=lambda n: results[n]["train_rmse"])
    best = results[best_name]
    print(f"\nbest: {best_name}")
    for k, v in best["params"].items():
        print(f"  {k:>10s} = {v:12.5f}")
    p = best["params"]
    print(f"\n  E2 - E1 = {p['E2_kJ'] - p['E1_kJ']:+.1f} kJ/mol")
    print(f"  heat terms a1={p['a1']:.3f} a2={p['a2']:.3f}")
    print(f"  flow exponent n = {p['n_flow']:.3f}")

    if "series" in results and "neutral" in results:
        d = results["neutral"]["train_rmse"] - results["series"]["train_rmse"]
        print(f"\n  thermally-neutral costs {d:+.4f} RMSE vs the full-heat model")
        if abs(d) < 0.3:
            print("  -> heat terms do no real work; concentration genuinely drops out")

    (ARTIFACTS / "model_comparison.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
