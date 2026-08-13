"""Does plug flow actually hold? Sweep a tanks-in-series model over N.

The problem statement says the reference simulation is a CFD / boundary value
problem, and a BVP is what you get when axial dispersion matters -- i.e. when
the reactor is *not* ideal plug flow. Tanks-in-series is the standard discrete
stand-in for dispersion: N equal CSTRs recover plug flow as N -> infinity and
approach a single well-mixed vessel at N = 1.

So the sweep is a real test rather than an extra knob. If error falls
monotonically with N, plug flow is right and dispersion is not the missing
mechanism. If some finite N wins clearly, the reactor is dispersed and the
fitted N is a physically meaningful Peclet-number surrogate.

N is swept over a fixed grid rather than fitted: rounding a continuous N would
make the objective discontinuous and wreck the least-squares Jacobian.
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
from src.physics import LOWER, PARAM_NAMES, Q_REF, UPPER, ReactorParams, _rate

ARTIFACTS = ROOT / "artifacts"
# n_flow is pinned to 0 here: the full comparison showed it lands at ~ -0.03.
FREE = [i for i, n in enumerate(PARAM_NAMES) if n != "n_flow"]


def integrate_tis(x, inputs, n_tanks: int, n_iter: int = 40) -> np.ndarray:
    """Steady-state cascade of `n_tanks` equal CSTRs.

    Each tank's balances are implicit in its own outlet temperature (the rate
    constants depend on it), so each is solved by fixed-point iteration --
    cheap, and it converges quickly because the jacket term is contracting.
    """
    p = ReactorParams.from_vector(x)
    CA0, T_jacket, tau = inputs["CA0"], inputs["T_jacket"], inputs["tau"]
    U = p.U if p.n_flow == 0.0 else p.U * (inputs["Q"] / Q_REF) ** p.n_flow
    theta = tau / n_tanks

    xA = np.ones_like(CA0)
    xB = np.zeros_like(CA0)
    T = inputs["T_in"].copy()

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(n_tanks):
            xA_in, xB_in, T_in = xA, xB, T
            T_out = T_in.copy()
            for _ in range(n_iter):
                k1 = _rate(p.ln_k1_ref, p.E1_kJ, T_out)
                k2 = _rate(p.ln_k2_ref, p.E2_kJ, T_out)
                xA_out = xA_in / (1.0 + k1 * theta)
                xB_out = (xB_in + k1 * theta * xA_out) / (1.0 + k2 * theta)
                heat = p.a1 * CA0 * k1 * xA_out + p.a2 * CA0 * k2 * xB_out
                T_new = (T_in + theta * heat + theta * U * T_jacket) / (1.0 + theta * U)
                if np.nanmax(np.abs(T_new - T_out)) < 1e-9:
                    T_out = T_new
                    break
                T_out = T_new
            xA, xB, T = xA_out, xB_out, T_out

    y = 100.0 * xB
    return np.where(np.isfinite(y), np.clip(y, 0.0, 100.0), 0.0)


def tis_residuals(x_free, n_tanks, inputs, y):
    full = np.zeros(len(PARAM_NAMES))
    full[FREE] = x_free
    r = integrate_tis(full, inputs, n_tanks) - y
    return np.where(np.isfinite(r), r, 1e3)


def tis_objective(x_free, n_tanks, inputs, y):
    return float(np.sqrt(np.mean(tis_residuals(x_free, n_tanks, inputs, y) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tanks", nargs="*", type=int, default=[1, 2, 3, 5, 8, 16, 32])
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--popsize", type=int, default=18)
    ap.add_argument("--polish-starts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_train()
    inputs = ode_inputs(df)
    y = df[TARGET].to_numpy(dtype=float)
    lo, hi = LOWER[FREE], UPPER[FREE]

    results = {}
    for n in args.tanks:
        t0 = time.perf_counter()
        de = differential_evolution(
            tis_objective, bounds=list(zip(lo, hi)), args=(n, inputs, y),
            maxiter=args.maxiter, popsize=args.popsize, tol=1e-9,
            mutation=(0.3, 1.2), recombination=0.85,
            seed=args.seed, polish=False, init="sobol",
            updating="deferred", workers=-1,
        )
        rng = np.random.default_rng(args.seed)
        span = hi - lo
        best_x, best = de.x, float(de.fun)
        for k in range(args.polish_starts):
            x0 = de.x if k == 0 else np.clip(de.x + rng.normal(0, 0.03, de.x.shape) * span, lo, hi)
            try:
                res = least_squares(tis_residuals, x0=x0, bounds=(lo, hi), args=(n, inputs, y),
                                    x_scale="jac", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=4000)
            except Exception:
                continue
            r = rmse(y, integrate_tis(np.pad(res.x, (0, 1)), inputs, n))
            if r < best:
                best, best_x = r, res.x
        full = np.zeros(len(PARAM_NAMES)); full[FREE] = best_x
        results[n] = {"train_rmse": best, "vector": [float(v) for v in full]}
        print(f"N={n:3d}  train RMSE {best:.4f}   ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"\n{'N tanks':>8s} {'train RMSE':>11s}")
    for n in args.tanks:
        print(f"{n:8d} {results[n]['train_rmse']:11.4f}")

    best_n = min(results, key=lambda n: results[n]["train_rmse"])
    print(f"\nbest N = {best_n} at RMSE {results[best_n]['train_rmse']:.4f}")

    # The comparison that matters is against the N -> infinity (plug flow)
    # limit, not against the rest of the grid. "Best in grid" says nothing on
    # its own: error can fall monotonically across every N swept and still be
    # worse than plug flow everywhere.
    pfr_path = ARTIFACTS / "physics_params.json"
    if pfr_path.exists():
        pfr_rmse = json.loads(pfr_path.read_text())["train_rmse"]
        print(f"plug-flow limit (N -> inf): {pfr_rmse:.4f}")
        if results[best_n]["train_rmse"] < pfr_rmse - 0.05:
            print(f"  -> N={best_n} beats plug flow: the reactor is genuinely dispersed.")
            if best_n == max(args.tanks):
                print("     Still falling at the largest N swept -- extend the grid.")
        else:
            print("  -> no finite N beats plug flow; dispersion is not the missing mechanism.")

    (ARTIFACTS / "tanks_sweep.json").write_text(json.dumps(
        {str(k): v for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
