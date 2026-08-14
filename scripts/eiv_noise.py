"""Errors-in-variables: MEASURE the input noise instead of inferring it.

The earlier argument for a noise floor was magnitude-matching -- inject N(0, 2.24 K) and
observe that it reproduces the residual scale. That is weak. It also leaned on a
zero-mean-residual test which is close to vacuous: least squares drives the residual
orthogonal to d f/d theta, so small mean residual is a first-order condition of the
optimizer, not evidence about noise.

This measures it. For each row we solve for the smallest temperature offset Delta_i that
makes the model reproduce the observed yield *exactly*:

    f(x_i + Delta_i ; theta) = y_i

The spread of the fitted Delta_i is then a direct estimate of the input-noise scale, and --
more importantly -- the Delta_i can be scanned for structure. Genuine measurement noise is
structureless; a misspecified model produces Delta_i that correlate with the features.

Three variants are fitted (offset on both temperatures, inlet only, jacket only) to see
where the noise sits.

CRITICAL SIDE CHECK: ordinary least squares with noisy predictors attenuates parameter
estimates. If the errors-in-variables optimum moves E2 away from 250.07, that shift
propagates straight into the cliff rows, which depend steeply on E2.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, add_physics_features, load_train, ode_inputs, rmse
from src.physics import SUBMIT_STEPS, integrate

ARTIFACTS = ROOT / "artifacts"
STEPS = 512


def _shift(df, delta, mode):
    d = df.copy()
    if mode in ("both", "inlet"):
        d["inlet_temperature_K"] = d["inlet_temperature_K"] + delta
    if mode in ("both", "jacket"):
        d["jacket_temperature_K"] = d["jacket_temperature_K"] + delta
    return d


def solve_offsets(x, df, y, mode="both", lo=-25.0, hi=25.0, n_grid=121, n_bisect=40):
    """Smallest per-row temperature offset reproducing y exactly (NaN if unreachable).

    Yield is not globally monotone in temperature (there is an interior optimum), so we
    first scan a grid to find sign changes of f - y, then take the bracket whose midpoint
    is closest to zero offset and bisect inside it.
    """
    grid = np.linspace(lo, hi, n_grid)
    F = np.empty((n_grid, len(df)))
    for i, g in enumerate(grid):
        F[i] = integrate(x, ode_inputs(_shift(df, g, mode)), n_steps=STEPS)
    D = F - y[None, :]

    # A row the model ALREADY reproduces has no sign change (f - y is ~0 across a wide
    # band), so a strict sign-change search would mislabel it "unreachable" -- which would
    # silently drop every well-fit row, including the 37 exact zeros, and bias the whole
    # estimate toward the hard rows. Accept any grid point where |f - y| is already
    # negligible, taking the one with the SMALLEST |offset|.
    TOL = 0.05
    best_lo = np.full(len(df), np.nan)
    best_hi = np.full(len(df), np.nan)
    direct = np.full(len(df), np.nan)
    sign_change = (np.sign(D[:-1]) * np.sign(D[1:])) < 0
    for j in range(len(df)):
        near = np.where(np.abs(D[:, j]) < TOL)[0]
        if len(near):
            direct[j] = grid[near[int(np.argmin(np.abs(grid[near])))]]
            continue
        idx = np.where(sign_change[:, j])[0]
        if len(idx) == 0:
            continue
        mids = 0.5 * (grid[idx] + grid[idx + 1])
        k = idx[int(np.argmin(np.abs(mids)))]
        best_lo[j], best_hi[j] = grid[k], grid[k + 1]

    ok = ~np.isnan(best_lo)
    a, b = best_lo.copy(), best_hi.copy()
    a[~ok], b[~ok] = 0.0, 0.0
    for _ in range(n_bisect):
        m = 0.5 * (a + b)
        fm = integrate(x, ode_inputs(_shift(df, m, mode)), n_steps=STEPS) - y
        fa = integrate(x, ode_inputs(_shift(df, a, mode)), n_steps=STEPS) - y
        same = np.sign(fm) == np.sign(fa)
        a = np.where(same, m, a)
        b = np.where(same, b, m)
    delta = 0.5 * (a + b)
    delta[~ok] = np.nan
    # Rows already reproduced without any offset take their direct (near-zero) solution.
    solved_direct = ~np.isnan(direct)
    delta = np.where(solved_direct, direct, delta)
    ok = ok | solved_direct
    return delta, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["both", "inlet", "jacket"])
    args = ap.parse_args()

    tr = load_train()
    y = tr[TARGET].to_numpy(dtype=float)
    x = json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"]
    base_rmse = rmse(y, integrate(x, ode_inputs(tr), n_steps=SUBMIT_STEPS))
    print(f"baseline train RMSE {base_rmse:.4f}\n")

    out = {}
    for mode in args.modes:
        delta, ok = solve_offsets(x, tr, y, mode=mode)
        d = delta[ok]
        # Rows the model cannot reach at ANY offset in +-25 K are model-form failures,
        # not noise -- report them separately rather than hiding them.
        print(f"--- offset applied to: {mode} ---")
        print(f"  rows explained exactly: {ok.sum()}/{len(tr)}   unreachable: {(~ok).sum()}")
        print(f"  Delta: mean {d.mean():+.3f} K, std {d.std(ddof=1):.3f} K, "
              f"median |Delta| {np.median(np.abs(d)):.3f} K, max |Delta| {np.abs(d).max():.2f} K")
        pct = np.percentile(np.abs(d), [50, 90, 99])
        print(f"  |Delta| percentiles  50%: {pct[0]:.2f}   90%: {pct[1]:.2f}   99%: {pct[2]:.2f} K")
        out[mode] = {"n_ok": int(ok.sum()), "std": float(d.std(ddof=1)),
                     "mean": float(d.mean()), "delta": [float(v) for v in delta]}

    # Structure scan on the fitted offsets: noise is structureless, model error is not.
    best = args.modes[0]
    delta = np.array(out[best]["delta"])
    ok = ~np.isnan(delta)
    f = add_physics_features(tr)
    cols = ["flow_rate_L_min", "concentration_mol_L", "inlet_temperature_K", "length_m",
            "jacket_temperature_K", "tau", "log_tau", "T_avg", "delta_T", "inv_T_avg"]
    print(f"\n--- structure scan on Delta ({best}), n={ok.sum()} ---")
    rows = []
    for c in cols:
        r = np.corrcoef(delta[ok], f[c].to_numpy()[ok])[0, 1]
        rows.append((abs(r), c, r))
    for a, c, r in sorted(rows, reverse=True):
        flag = "  <-- STRUCTURE" if a > 0.25 else ""
        print(f"  corr(Delta, {c:22s}) = {r:+.3f}{flag}")
    largest = max(rows)[0]
    print(f"\n  largest |corr| = {largest:.3f}")
    print("  -> Delta looks like structureless noise" if largest <= 0.25 else
          "  -> Delta carries STRUCTURE: this is misspecification, not pure noise")

    (ARTIFACTS / "eiv_offsets.json").write_text(json.dumps(
        {"baseline_train_rmse": base_rmse, "modes": out,
         "structure_max_abs_corr": float(largest)}, indent=2))
    print(f"\nwrote {ARTIFACTS / 'eiv_offsets.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
