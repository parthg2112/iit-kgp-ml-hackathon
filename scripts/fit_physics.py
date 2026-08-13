"""Recover the seven reactor parameters by multistart nonlinear least squares.

Strategy: a global differential-evolution sweep at the cheaper step count to
find the basin, then a least_squares polish at full accuracy. The surface has
local minima (a fast-k1/fast-k2 pair can mimic a slow/slow pair over a limited
tau range), so the global stage is not optional.

Go/no-go gate: the target is deterministic simulator output, so a *correct*
functional form should drive train RMSE toward ~0, not toward 8. A fit that
plateaus in the 6-10 band means the model form is wrong -- more multistarts
will not rescue it.

Usage:
    python scripts/fit_physics.py [--maxiter N] [--seed S] [--tag NAME]
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
    CONVERGENCE_TOL,
    DEFAULT_STEPS,
    LOWER,
    PARAM_NAMES,
    SEARCH_STEPS,
    UPPER,
    ReactorParams,
    integrate,
    objective,
    residuals,
)

ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--popsize", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--polish-starts", type=int, default=12)
    ap.add_argument("--tag", type=str, default="physics_params")
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    df = load_train()
    inputs = ode_inputs(df)
    y = df[TARGET].to_numpy(dtype=float)

    bounds = list(zip(LOWER, UPPER))
    t0 = time.perf_counter()

    print(f"global search: popsize={args.popsize} maxiter={args.maxiter} steps={SEARCH_STEPS}")
    de = differential_evolution(
        objective,
        bounds=bounds,
        # Guard ON for the global search: it keeps DE out of stiff corners where
        # the integration has not converged. It is deliberately left OFF for the
        # polish below (residuals() defaults it off) -- a poison value is a cliff
        # in the objective and ruins a finite-difference Jacobian.
        args=(inputs, y, SEARCH_STEPS, CONVERGENCE_TOL),
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=1e-8,
        mutation=(0.3, 1.2),
        recombination=0.85,
        seed=args.seed,
        polish=False,
        init="sobol",
        updating="deferred",
        workers=-1,
    )
    print(f"  DE best RMSE {de.fun:.4f}  ({time.perf_counter() - t0:.0f}s, {de.nfev} evals)")

    # Polish from the DE optimum plus perturbed restarts, at full step count.
    rng = np.random.default_rng(args.seed)
    starts = [de.x]
    span = UPPER - LOWER
    for _ in range(args.polish_starts - 1):
        jitter = de.x + rng.normal(0.0, 0.04, size=de.x.shape) * span
        starts.append(np.clip(jitter, LOWER, UPPER))

    best_x, best_rmse = de.x, float(de.fun)
    for i, x0 in enumerate(starts):
        try:
            res = least_squares(
                residuals,
                x0=x0,
                bounds=(LOWER, UPPER),
                args=(inputs, y, DEFAULT_STEPS),
                x_scale="jac",
                loss="linear",
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                max_nfev=8000,
            )
        except Exception as exc:  # a diverged start should not kill the run
            print(f"  polish {i:2d}: failed ({exc})")
            continue
        r = rmse(y, integrate(res.x, inputs, n_steps=DEFAULT_STEPS))
        marker = ""
        if r < best_rmse:
            best_rmse, best_x = r, res.x
            marker = "  <-- best"
        print(f"  polish {i:2d}: RMSE {r:.4f}{marker}")

    p = ReactorParams.from_vector(best_x)
    pred = integrate(best_x, inputs, n_steps=DEFAULT_STEPS)

    # Convergence must hold at the parameters we actually ship, not just at the
    # synthetic sets in check_integrator.py.
    fine = integrate(best_x, inputs, n_steps=4 * DEFAULT_STEPS)
    step_drift = float(np.abs(pred - fine).max())

    elapsed = time.perf_counter() - t0
    print(f"\nbest train RMSE: {best_rmse:.4f}   ({elapsed:.0f}s total)")
    print(f"step convergence at fitted params (x4 substeps): {step_drift:.3e}")
    for name in PARAM_NAMES:
        print(f"  {name:>10s} = {getattr(p, name):12.5f}")

    k_ratio = np.exp(p.ln_k2_ref - p.ln_k1_ref)
    print(f"\n  E2 - E1 = {p.E2_kJ - p.E1_kJ:+.1f} kJ/mol"
          f"   ({'E2 > E1 as predicted' if p.E2_kJ > p.E1_kJ else 'E2 < E1 -- unexpected'})")
    print(f"  k2/k1 at T_ref = {k_ratio:.4f}")
    print(f"  heat terms a1={p.a1:.3f} a2={p.a2:.3f}"
          f"   ({'near thermally neutral' if max(abs(p.a1), abs(p.a2)) < 1.0 else 'significant heat release'})")

    if best_rmse < 1.0:
        verdict = "PASS — functional form looks correct, ship physics-dominant"
    elif best_rmse < 6.0:
        verdict = "PARTIAL — better than the tree baseline but form may be incomplete"
    else:
        verdict = "FAIL — form is likely wrong; try the A->C parallel path or axial dispersion"
    print(f"\ngate: {verdict}")

    out = ARTIFACTS / f"{args.tag}.json"
    out.write_text(json.dumps({
        "params": p.as_dict(),
        "vector": [float(v) for v in best_x],
        "train_rmse": best_rmse,
        "de_rmse": float(de.fun),
        "step_drift": step_drift,
        "seed": args.seed,
        "maxiter": args.maxiter,
        "n_steps": DEFAULT_STEPS,
        "elapsed_s": elapsed,
    }, indent=2))
    np.save(ARTIFACTS / f"{args.tag}_train_pred.npy", pred)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
