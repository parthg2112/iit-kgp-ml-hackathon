"""Should we average over parameter uncertainty instead of shipping a point estimate?

Several distinct parameter sets fit the training data about equally well (the multistart
optima, the model variants, and the cold-fold refits). Under squared error, averaging over
genuine parameter uncertainty is usually better than picking one -- but only if the
uncertainty is real rather than an artifact of some fits simply being worse.

Measured spread on the test set: 44/50 rows move by less than 2 across these sets, so the
question is decided almost entirely by rows 39 and 24.

Validation is out-of-fold: build the same ensemble inside each CV fold and compare against
the single-fit predictions on the held-out rows. Adopt only on a positive result.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_test, load_train, ode_inputs, rmse
from src.physics import DEFAULT_STEPS, LOWER, SUBMIT_STEPS, UPPER, integrate, residuals

ARTIFACTS = ROOT / "artifacts"
TOLERANCE = 1.35  # keep parameter sets within 35% of the best train RMSE


def collect_vectors():
    """Every parameter set we have on disk that plausibly fits."""
    out = {}
    p = json.loads((ARTIFACTS / "physics_params.json").read_text())
    out["shipped"] = p["vector"]
    mc = ARTIFACTS / "model_comparison.json"
    if mc.exists():
        for k, v in json.loads(mc.read_text()).items():
            out[k] = v["vector"]
    cf = ARTIFACTS / "cold_folds.json"
    if cf.exists():
        for i, v in enumerate(json.loads(cf.read_text())["fold_vectors"]):
            out[f"fold{i}"] = v
    wb = ARTIFACTS / "wide_bounds.json"
    if wb.exists():
        out["wide"] = json.loads(wb.read_text())["vector"]
    return out


def main() -> int:
    tr, te = load_train(), load_test()
    y = tr[TARGET].to_numpy(dtype=float)
    inp_tr, inp_te = ode_inputs(tr), ode_inputs(te)

    scored = []
    for name, v in collect_vectors().items():
        v = np.clip(np.array(v, dtype=float), LOWER, UPPER)
        scored.append((rmse(y, integrate(v, inp_tr, n_steps=DEFAULT_STEPS)), name, v))
    scored.sort()
    best = scored[0][0]
    keep = [(s, n, v) for s, n, v in scored if s <= best * TOLERANCE]
    print(f"best train RMSE {best:.4f}; keeping {len(keep)}/{len(scored)} sets within {TOLERANCE:g}x")
    for s, n, _ in keep:
        print(f"   {n:16s} {s:.4f}")

    M = np.array([np.clip(integrate(v, inp_te, n_steps=SUBMIT_STEPS), 0, 100) for _, _, v in keep])
    single = np.clip(integrate(keep[0][2], inp_te, n_steps=SUBMIT_STEPS), 0, 100)
    ens = M.mean(0)
    print(f"\ntest-set effect: mean |ensemble - single| = {np.abs(ens - single).mean():.3f}, "
          f"max {np.abs(ens - single).max():.3f}")
    o = np.argsort(-np.abs(ens - single))[:5]
    for i in o:
        print(f"   row {i:2d}: single {single[i]:6.2f} -> ensemble {ens[i]:6.2f} (sd {M[:, i].std():5.2f})")

    # --- Out-of-fold validation: does ensembling actually generalize better? ---
    print("\nout-of-fold validation (ensemble of fold-refits vs single fold-refit):")
    starts = [v for _, _, v in keep[: min(5, len(keep))]]
    for n_splits in (10,):
        for seed in (0, 1, 2):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            oof_s, oof_e = np.zeros(len(tr)), np.zeros(len(tr))
            for tr_idx, te_idx in kf.split(tr):
                sub, held = tr.iloc[tr_idx], tr.iloc[te_idx]
                inp_s, inp_h = ode_inputs(sub), ode_inputs(held)
                ysub = sub[TARGET].to_numpy(dtype=float)
                preds = []
                for x0 in starts:
                    r = least_squares(residuals, x0=x0, bounds=(LOWER, UPPER),
                                      args=(inp_s, ysub, DEFAULT_STEPS, None), x_scale="jac",
                                      xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
                    preds.append(np.clip(integrate(r.x, inp_h, n_steps=DEFAULT_STEPS), 0, 100))
                oof_s[te_idx] = preds[0]
                oof_e[te_idx] = np.mean(preds, axis=0)
            print(f"   seed {seed}: single {rmse(y, oof_s):.4f}  ensemble {rmse(y, oof_e):.4f}  "
                  f"gain {rmse(y, oof_s) - rmse(y, oof_e):+.4f}")

    (ARTIFACTS / "ensemble.json").write_text(json.dumps({
        "kept": [n for _, n, _ in keep],
        "test_pred_ensemble": [float(v) for v in ens],
        "test_pred_single": [float(v) for v in single],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
