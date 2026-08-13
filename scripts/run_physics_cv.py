"""Honest cross-validation of the physics model, plus the blend-weight search.

The 7 ODE parameters are refit inside every fold, so this scores the modelling
procedure rather than one lucky parameter set. The blend weight against the
ExtraTrees safety net is then chosen by minimizing out-of-fold RMSE.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import fit_predict
from src.data import ROOT, TARGET, load_train, rmse
from src.evaluate import best_blend_weight, make_physics_predict_fn, repeated_cv
from src.residual import make_hybrid_predict_fn

ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="physics_params")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--shrinkage", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0])
    args = ap.parse_args()

    df = load_train()
    y = df[TARGET].to_numpy(dtype=float)
    x = json.loads((ARTIFACTS / f"{args.params}.json").read_text())["vector"]
    seeds = tuple(args.seeds)

    print(f"repeated 10-fold CV, seeds={seeds}, params refit inside every fold\n")

    phys_mat, phys_stats = repeated_cv(make_physics_predict_fn(x), df, seeds=seeds)
    print(f"{'physics (ODE)':32s} {phys_stats['mean']:8.3f} +/- {phys_stats['std']:.3f}")

    tree_mat, tree_stats = repeated_cv(lambda tr, te, s: fit_predict(tr, te, s), df, seeds=seeds)
    print(f"{'ExtraTrees safety net':32s} {tree_stats['mean']:8.3f} +/- {tree_stats['std']:.3f}")

    # Does a residual correction on top of the physics model earn its place?
    hybrid_stats = {}
    for s in args.shrinkage:
        if s == 0.0:
            continue
        _, st = repeated_cv(make_hybrid_predict_fn(x, shrinkage=s), df, seeds=seeds)
        hybrid_stats[s] = st
        print(f"{'hybrid, shrinkage ' + str(s):32s} {st['mean']:8.3f} +/- {st['std']:.3f}")

    # Blend weight from out-of-fold predictions, averaged over seeds.
    phys_oof, tree_oof = phys_mat.mean(0), tree_mat.mean(0)
    w, w_rmse, grid, scores = best_blend_weight(y, phys_oof, tree_oof)
    print(f"\nblend search: best weight w={w:.3f} on physics -> OOF RMSE {w_rmse:.3f}")
    print(f"  w=1.000 (physics only)     {rmse(y, phys_oof):.3f}")
    print(f"  w=0.700 (the brief's 70/30) {rmse(y, np.clip(0.7 * phys_oof + 0.3 * tree_oof, 0, 100)):.3f}")
    print(f"  w=0.000 (tree only)        {rmse(y, tree_oof):.3f}")
    if w > 0.98:
        print("  -> the tree adds nothing; ship physics alone")

    np.save(ARTIFACTS / "physics_oof.npy", phys_mat)
    np.save(ARTIFACTS / "tree_oof.npy", tree_mat)
    (ARTIFACTS / "blend.json").write_text(json.dumps({
        "weight": w,
        "oof_rmse_at_weight": w_rmse,
        "physics_cv": phys_stats,
        "tree_cv": tree_stats,
        "hybrid_cv": {str(k): v for k, v in hybrid_stats.items()},
        "seeds": list(seeds),
    }, indent=2))
    print(f"\nwrote {ARTIFACTS / 'blend.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
