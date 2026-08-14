"""Noise-averaged prediction: predict E[f(x_true) | x_observed], not f(x_observed).

IF the hidden targets are f(true inputs) and the inputs we are given carry temperature
noise, then under squared loss the optimal prediction is the model averaged over the noise
distribution, not the model evaluated at the observed inputs. Where the response is curved
-- which is exactly the cliff -- those differ.

This is an ASSUMPTION, not an established fact (see the note in CLAUDE.md): it requires the
targets to have been generated from inputs that differ from the ones published. The
evidence for it is indirect. So the gate here is empirical: apply the same smoothing inside
cross-validation and adopt only if held-out error actually improves.

Integration is Gauss-Hermite over a common temperature offset applied to both inlet and
jacket -- the errors-in-variables fit explained the most rows (138/150) in that mode.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_test, load_train, ode_inputs, rmse
from src.physics import DEFAULT_STEPS, LOWER, SUBMIT_STEPS, UPPER, integrate, residuals

ARTIFACTS = ROOT / "artifacts"


def smoothed(x, df, sigma, n_nodes=7, n_steps=DEFAULT_STEPS):
    """E[f(x + delta)] with delta ~ N(0, sigma^2) on both temperatures, by Gauss-Hermite."""
    if sigma <= 0:
        return integrate(x, ode_inputs(df), n_steps=n_steps)
    nodes, weights = hermegauss(n_nodes)          # weight function exp(-t^2/2)
    weights = weights / weights.sum()
    acc = np.zeros(len(df))
    for t, w in zip(nodes, weights):
        d = df.copy()
        d["inlet_temperature_K"] = d["inlet_temperature_K"] + sigma * t
        d["jacket_temperature_K"] = d["jacket_temperature_K"] + sigma * t
        acc += w * integrate(x, ode_inputs(d), n_steps=n_steps)
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", nargs="*", type=float, default=[0.0, 1.0, 1.7, 2.5, 4.0])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    tr, te = load_train(), load_test()
    y = tr[TARGET].to_numpy(dtype=float)
    x = np.array(json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"])

    print("in-sample effect of smoothing (train RMSE):")
    for s in args.sigmas:
        print(f"  sigma {s:4.1f} K -> {rmse(y, smoothed(x, tr, s, n_steps=SUBMIT_STEPS)):.4f}")

    # --- The gate: does smoothing help OUT of sample? Parameters refit per fold, then
    #     the same smoothing applied to the held-out rows.
    print("\n10-fold CV with smoothing applied to held-out predictions:")
    print(f"{'sigma':>7s} " + "".join(f"{'seed ' + str(s):>10s}" for s in args.seeds) + f"{'mean':>10s}")
    results = {}
    fold_params = {}
    for seed in args.seeds:
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        fold_params[seed] = []
        for tr_idx, te_idx in kf.split(tr):
            sub = tr.iloc[tr_idx]
            r = least_squares(residuals, x0=x, bounds=(LOWER, UPPER),
                              args=(ode_inputs(sub), sub[TARGET].to_numpy(float), DEFAULT_STEPS, None),
                              x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
            fold_params[seed].append((r.x, te_idx))

    for s in args.sigmas:
        per_seed = []
        for seed in args.seeds:
            oof = np.zeros(len(tr))
            for xf, te_idx in fold_params[seed]:
                oof[te_idx] = smoothed(xf, tr.iloc[te_idx], s)
            per_seed.append(rmse(y, oof))
        results[s] = per_seed
        print(f"{s:7.1f} " + "".join(f"{v:10.4f}" for v in per_seed) + f"{np.mean(per_seed):10.4f}")

    base = np.mean(results[0.0])
    best_s = min(results, key=lambda k: np.mean(results[k]))
    gain = base - np.mean(results[best_s])
    consistent = all(results[best_s][i] <= results[0.0][i] for i in range(len(args.seeds)))
    print(f"\nbest sigma {best_s:.1f} K: CV {np.mean(results[best_s]):.4f} vs unsmoothed "
          f"{base:.4f}  (gain {gain:+.4f}, better on all seeds: {consistent})")
    adopt = best_s > 0 and gain > 0.05 and consistent
    print("VERDICT:", "ADOPT noise-averaged prediction" if adopt else
          "REJECT -- smoothing does not improve held-out error")

    out = {"cv": {str(k): v for k, v in results.items()}, "best_sigma": float(best_s),
           "gain": float(gain), "consistent": bool(consistent), "adopt": bool(adopt)}

    if adopt:
        cur = np.array(json.loads((ARTIFACTS / "blend.json").read_text())["weight"])
        sm = np.clip(smoothed(x, te, best_s, n_steps=SUBMIT_STEPS), 0, 100)
        raw = np.clip(integrate(x, ode_inputs(te), n_steps=SUBMIT_STEPS), 0, 100)
        d = sm - raw
        print(f"\nper-row effect on the test set (physics component only):")
        print(f"  mean |delta| {np.abs(d).mean():.3f}, max {np.abs(d).max():.3f}")
        for i in np.argsort(-np.abs(d))[:8]:
            flag = "  <-- >5 points" if abs(d[i]) > 5 else ""
            print(f"    row {i:2d}: {raw[i]:7.3f} -> {sm[i]:7.3f}  ({d[i]:+.3f}){flag}")
        out["test_smoothed"] = [float(v) for v in sm]
        out["test_raw"] = [float(v) for v in raw]

    (ARTIFACTS / "noise_averaged.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
