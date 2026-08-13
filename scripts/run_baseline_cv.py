"""Repeated 10-fold CV for the ExtraTrees safety net.

Also records the raw-feature variants so the "physics features cut error by
27%" claim is reproduced here rather than inherited from the team brief.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import fit_predict
from src.data import FEATURE_COLUMNS, RAW_FEATURES, ROOT, TARGET, add_physics_features, load_train
from src.evaluate import repeated_cv

ARTIFACTS = ROOT / "artifacts"
SEEDS = (0, 1, 2, 3, 4)


def make_fn(model_cls, columns, **kw):
    def fn(train_df, predict_df, seed):
        model = model_cls(random_state=seed, **kw)
        Xtr = add_physics_features(train_df)[columns].to_numpy(dtype=float)
        model.fit(Xtr, train_df[TARGET].to_numpy(dtype=float))
        Xte = add_physics_features(predict_df)[columns].to_numpy(dtype=float)
        return np.clip(model.predict(Xte), 0.0, 100.0)

    return fn


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_train()

    variants = {
        "ExtraTrees raw": make_fn(ExtraTreesRegressor, RAW_FEATURES, n_estimators=800, n_jobs=-1),
        "RandomForest raw": make_fn(RandomForestRegressor, RAW_FEATURES, n_estimators=800, n_jobs=-1),
        "GradientBoosting raw": make_fn(GradientBoostingRegressor, RAW_FEATURES),
        "RandomForest physics": make_fn(RandomForestRegressor, FEATURE_COLUMNS, n_estimators=800, n_jobs=-1),
        "GradientBoosting physics": make_fn(GradientBoostingRegressor, FEATURE_COLUMNS),
        "ExtraTrees physics": lambda tr, te, seed: fit_predict(tr, te, seed),
    }

    print(f"repeated 10-fold CV over seeds {SEEDS}\n")
    print(f"{'model':28s} {'CV RMSE':>16s}")
    results = {}
    for name, fn in variants.items():
        mat, stats = repeated_cv(fn, df, seeds=SEEDS)
        results[name] = stats
        print(f"{name:28s} {stats['mean']:8.3f} +/- {stats['std']:.3f}")
        if name == "ExtraTrees physics":
            np.save(ARTIFACTS / "baseline_oof.npy", mat)

    (ARTIFACTS / "baseline_cv.json").write_text(json.dumps(results, indent=2))

    et_raw = results["ExtraTrees raw"]["mean"]
    et_phys = results["ExtraTrees physics"]["mean"]
    print(f"\nphysics features change ExtraTrees error by {100 * (et_phys - et_raw) / et_raw:+.1f}%")
    print(f"(team brief claimed 19.2 -> 14.0, i.e. -27%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
