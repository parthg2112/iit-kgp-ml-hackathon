"""Re-verify the audit's headline recommendation: physics + ML residual correction.

An external audit reported physics-only at CV 10.83 and physics+residual at 7.57, and
concluded the residual stage is a 49% win. Our physics fit is ~3x better than theirs
(train RMSE 3.66 vs 11.3), so the question is whether a residual corrector still adds
anything once the physics is already good -- a much harder bar.

An earlier sweep here found gains <= 0.05, but only over shrinkage on a single corrector
type. This retests properly: three corrector families, multiple seeds, ODE parameters
refit inside every fold, so the whole pipeline is what gets cross-validated.

Gate: adopt only if a corrector beats physics-only by > 0.3 out-of-sample.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_train
from src.evaluate import make_physics_predict_fn, repeated_cv
from src.residual import make_hybrid_predict_fn

ARTIFACTS = ROOT / "artifacts"
SEEDS = (0, 1, 2, 3, 4)


def ridge(seed):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def shallow_et(seed):
    return ExtraTreesRegressor(n_estimators=600, max_depth=5, min_samples_leaf=8,
                               max_features=0.5, bootstrap=False, random_state=seed, n_jobs=-1)


def small_gbm(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03,
                                     subsample=0.8, random_state=seed)


CORRECTORS = {"ridge": ridge, "shallow_extratrees": shallow_et, "small_gbm": small_gbm}


def main() -> int:
    df = load_train()
    x = json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"]
    print(f"repeated 10-fold CV, seeds={SEEDS}, ODE refit inside every fold\n")

    _, base = repeated_cv(make_physics_predict_fn(x), df, seeds=SEEDS)
    print(f"{'physics only (baseline)':34s} {base['mean']:7.3f} +/- {base['std']:.3f}")

    results = {"physics_only": base}
    for name, factory in CORRECTORS.items():
        for shrink in (0.5, 1.0):
            _, st = repeated_cv(
                make_hybrid_predict_fn(x, shrinkage=shrink, corrector_factory=factory),
                df, seeds=SEEDS,
            )
            key = f"{name}@{shrink}"
            results[key] = st
            delta = base["mean"] - st["mean"]
            print(f"{key:34s} {st['mean']:7.3f} +/- {st['std']:.3f}   vs physics {delta:+.3f}")

    best = min((k for k in results if k != "physics_only"), key=lambda k: results[k]["mean"])
    gain = base["mean"] - results[best]["mean"]
    print(f"\nbest corrector: {best}, gain {gain:+.3f} RMSE")
    if gain > 0.3:
        print("VERDICT: adopt -- residual correction earns its place.")
    else:
        print("VERDICT: reject -- gain is below the 0.3 bar.")
        print("  Our physics residual carries little learnable structure, which is exactly")
        print("  why the audit saw a large gain and we do not: their residual still held")
        print("  the model-form error our fit already captures.")

    (ARTIFACTS / "residual_verification.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
