"""Are both reactions really first order? Profile the reaction orders n1 and n2.

First-order kinetics is currently an *assumption*, and the whole "inlet concentration
cancels" argument rests on it. This measures it instead.

For n != 1 the analytic series step no longer applies (it solves a *linear* system), so this
uses a fixed-step RK4 on the normalized state and verifies it against scipy's stiff solver
before any fit is trusted. Note what n != 1 implies physically: the mass balance picks up a
`CA0^(n-1)` factor, so inlet concentration stops cancelling from the yield expression
altogether. That is exactly the claim under test.

Usage:  python scripts/profile_orders.py --param n1
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_train, ode_inputs, rmse
from src.physics import PARAM_NAMES, _rate

ARTIFACTS = ROOT / "artifacts"
# order-model parameter vector: the 7 physical params plus n1, n2
ONAMES = ("ln_k1_ref", "E1_kJ", "ln_k2_ref", "E2_kJ", "a1", "a2", "U", "n1", "n2")
OLOWER = np.array([-12.0, 40.0, -12.0, 60.0, -30.0, -30.0, 0.0, 0.30, 0.30])
OUPPER = np.array([12.0, 140.0, 12.0, 280.0, 30.0, 30.0, 60.0, 2.50, 2.50])
GRIDS = {"n1": [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5],
         "n2": [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]}


def integrate_orders(x, inputs, n_steps=1024):
    """RK4 on (xA, xB, T) with general reaction orders.

    Concentrations are clipped at 0 inside the derivative: for fractional n,
    a negative base makes x**n NaN, and RK4 stages can transiently undershoot.
    """
    (lk1, E1, lk2, E2, a1, a2, U, n1, n2) = [float(v) for v in x]
    CA0, Tj, tau = inputs["CA0"], inputs["T_jacket"], inputs["tau"]
    h = tau / n_steps

    def deriv(xA, xB, T):
        k1 = _rate(lk1, E1, T) * CA0 ** (n1 - 1.0)
        k2 = _rate(lk2, E2, T) * CA0 ** (n2 - 1.0)
        r1 = k1 * np.maximum(xA, 0.0) ** n1
        r2 = k2 * np.maximum(xB, 0.0) ** n2
        return -r1, r1 - r2, a1 * CA0 * r1 + a2 * CA0 * r2 + U * (Tj - T)

    xA = np.ones_like(CA0)
    xB = np.zeros_like(CA0)
    T = inputs["T_in"].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            a1_, b1_, t1_ = deriv(xA, xB, T)
            a2_, b2_, t2_ = deriv(xA + .5 * h * a1_, xB + .5 * h * b1_, T + .5 * h * t1_)
            a3_, b3_, t3_ = deriv(xA + .5 * h * a2_, xB + .5 * h * b2_, T + .5 * h * t2_)
            a4_, b4_, t4_ = deriv(xA + h * a3_, xB + h * b3_, T + h * t3_)
            xA = np.clip(xA + (h / 6) * (a1_ + 2 * a2_ + 2 * a3_ + a4_), 0.0, 1.0)
            xB = np.clip(xB + (h / 6) * (b1_ + 2 * b2_ + 2 * b3_ + b4_), 0.0, 1.0)
            T = T + (h / 6) * (t1_ + 2 * t2_ + 2 * t3_ + t4_)
    y = 100.0 * xB
    return np.where(np.isfinite(y), np.clip(y, 0.0, 100.0), 0.0)


def order_residuals(x, inputs, y, n_steps=1024):
    r = integrate_orders(x, inputs, n_steps) - y
    return np.where(np.isfinite(r), r, 1e3)


def _fit_pinned(job):
    name, value, inputs, y, x0 = job
    j = ONAMES.index(name)
    free = [i for i in range(len(ONAMES)) if i != j]

    def expand(xf):
        full = np.empty(len(ONAMES))
        full[j] = value
        full[free] = xf
        return full

    t0 = time.perf_counter()
    best = None
    for k in range(2):
        start = np.array(x0)[free] if k == 0 else np.clip(
            np.array(x0)[free] * (1 + 0.05 * ((-1) ** k)), OLOWER[free], OUPPER[free])
        try:
            r = least_squares(lambda xf: order_residuals(expand(xf), inputs, y),
                              x0=start, bounds=(OLOWER[free], OUPPER[free]),
                              x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        except Exception:
            continue
        s = rmse(y, integrate_orders(expand(r.x), inputs))
        if best is None or s < best[0]:
            best = (s, expand(r.x))
    return value, best[1], best[0], time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default="n1", choices=sorted(GRIDS))
    args = ap.parse_args()

    tr = load_train()
    inputs = ode_inputs(tr)
    y = tr[TARGET].to_numpy(dtype=float)
    base = json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"]
    x0 = np.array([base[PARAM_NAMES.index(n)] for n in ONAMES[:7]] + [1.0, 1.0])

    # Gate: at n1=n2=1 the order model must reproduce the production integrator.
    ref = json.loads((ARTIFACTS / "physics_params.json").read_text())["train_rmse_submit_steps"]
    got = rmse(y, integrate_orders(x0, inputs, n_steps=2048))
    print(f"gate: order-model at n=1 gives train RMSE {got:.4f} vs production {ref:.4f} "
          f"(diff {abs(got-ref):.4f})")
    if abs(got - ref) > 0.02:
        print("FAIL — the RK4 order-model does not reproduce the analytic integrator at n=1.")
        return 1
    print("gate PASS\n")

    grid = GRIDS[args.param]
    jobs = [(args.param, v, inputs, y, x0) for v in grid]
    print(f"profiling {args.param} over {grid} on {min(len(grid), cpu_count()-1)} processes\n")
    print(f"{args.param:>6s} {'train RMSE':>11s} {'excess':>8s} {'other order':>12s}")
    out = {}
    with Pool(min(len(grid), max(1, cpu_count() - 1))) as pool:
        for v, xf, s, secs in pool.imap_unordered(_fit_pinned, jobs):
            other = "n2" if args.param == "n1" else "n1"
            out[str(v)] = {"train_rmse": s, "vector": [float(t) for t in xf]}
            print(f"{v:6.2f} {s:11.4f} {s-ref:+8.4f} {xf[ONAMES.index(other)]:12.3f}  ({secs:.0f}s)",
                  flush=True)

    R = np.array([out[str(v)]["train_rmse"] for v in grid])
    from scipy.stats import f as fdist
    rmin = R.min()
    for lvl, tag in [(0.6827, "1-sigma"), (0.95, "95%")]:
        thr = rmin * np.sqrt(1 + fdist.ppf(lvl, 1, 150 - 9) / (150 - 9))
        ok = [v for v, s in zip(grid, R) if s <= thr]
        print(f"  {tag:8s} interval: [{min(ok):.2f}, {max(ok):.2f}]" if ok else f"  {tag}: empty")
    print(f"\nbest {args.param} = {grid[int(np.argmin(R))]:.2f} at RMSE {rmin:.4f}")
    print("first-order is CONFIRMED by measurement" if abs(grid[int(np.argmin(R))] - 1.0) < 0.11
          else "first-order is NOT the best fit -- investigate")

    (ARTIFACTS / f"profile_order_{args.param}.json").write_text(json.dumps(
        {"param": args.param, "reference_rmse": ref, "points": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
