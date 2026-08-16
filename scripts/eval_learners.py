"""Is the blend's second member the wrong MODEL, not the wrong features?

The feature experiment came back negative in a way that pointed somewhere else. Adding
reaction-engineering features made ExtraTrees monotonically worse, and replacing the feature
set with a small physical one was worse still -- so it is not dilution. But on the single
best feature, the analytic isothermal yield Yiso:

    linear fit  y ~ Yiso   -> RMSE 24.04 (in-sample)
    ExtraTrees  on Yiso    -> RMSE 33.99 (pooled OOF, 5 seeds)

A straight line beats the tree on the same column, because with min_samples_leaf=1 on one
feature ExtraTrees is effectively 1-NN. And Yiso is the strongest single feature we have:
within the alive rows (y >= 1) it correlates +0.669 against +0.458 for log_tau.

So the hypothesis under test here is: the feature is fine, the LEARNER is wrong. The yield
surface is smooth in the right coordinates, and a piecewise-constant estimator is the wrong
shape for it.

WHAT DECIDES IT is the blend, not standalone accuracy. A weak member with decorrelated error
is worth more than a strong member that fails on the same rows as the ODE:

    w* = (s2^2 - rho*s1*s2) / (s1^2 + s2^2 - 2*rho*s1*s2)

so rho is reported for every candidate and the ranking is by projected blend RMSE.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import FEATURE_COLUMNS, ROOT, TARGET, add_physics_features, load_train, rmse
from src.features import TRIAL_KINETICS, add_reaction_features, reaction_feature_columns
from src.physics import NOISE_SIGMA_K

ARTIFACTS = ROOT / "artifacts"
SEEDS = (0, 1, 2, 3, 4)
PHYS_SEEDS = 3
ONE = TRIAL_KINETICS[:1]


def _kr():
    """Kernel ridge with alpha/gamma chosen INSIDE each fold -- never on the scoring rows."""
    return make_pipeline(StandardScaler(), GridSearchCV(
        KernelRidge(kernel="rbf"),
        {"alpha": [1e-3, 1e-2, 1e-1, 1.0], "gamma": [0.01, 0.03, 0.1, 0.3, 1.0]},
        cv=5, scoring="neg_root_mean_squared_error"))


def _gp():
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(1)) + WhiteKernel(1.0)
    return make_pipeline(StandardScaler(),
                         GaussianProcessRegressor(kernel=k, normalize_y=True,
                                                  n_restarts_optimizer=2, random_state=0))


MODELS = {
    "ExtraTrees leaf=1 (shipped)": lambda s: ExtraTreesRegressor(
        n_estimators=800, max_features=0.6, min_samples_leaf=1, bootstrap=False,
        random_state=s, n_jobs=-1),
    "ExtraTrees leaf=4":           lambda s: ExtraTreesRegressor(
        n_estimators=800, max_features=0.6, min_samples_leaf=4, bootstrap=False,
        random_state=s, n_jobs=-1),
    "Ridge (linear)":              lambda s: make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40))),
    "KernelRidge RBF":             lambda s: _kr(),
    "GP (RBF + white)":            lambda s: _gp(),
    "kNN k=5 (scaled)":            lambda s: make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")),
}

FEATS = {
    "shipped 13":     (FEATURE_COLUMNS, add_physics_features),
    "reaction 19":    (reaction_feature_columns(ONE), lambda d: add_reaction_features(d, ONE)),
    "Yiso block 6":   ([f"{n}_0" for n in ("Yiso", "logtau_rel", "Yfrac",
                                           "lnDa1", "lnDa2", "ln_sel")],
                       lambda d: add_reaction_features(d, ONE)),
}


def oof(df, cols, fn, model_factory, seed):
    X = fn(df)[cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    out = np.zeros(len(df))
    for a, b in KFold(10, shuffle=True, random_state=seed).split(X):
        m = model_factory(seed)
        m.fit(X[a], y[a])
        out[b] = np.clip(m.predict(X[b]), 0.0, 100.0)
    return out


def optimal_blend(s1, s2, rho):
    c = rho * s1 * s2
    den = s1 ** 2 + s2 ** 2 - 2 * c
    w = 1.0 if den <= 0 else float(np.clip((s2 ** 2 - c) / den, 0.0, 1.0))
    return w, float(np.sqrt(w * w * s1 ** 2 + (1 - w) ** 2 * s2 ** 2 + 2 * w * (1 - w) * c))


def main() -> int:
    tr = load_train()
    y = tr[TARGET].to_numpy(dtype=float)
    phys = np.load(ARTIFACTS / "oof_by_sigma.npz")[f"s_{NOISE_SIGMA_K}"]
    s1 = float(np.mean([rmse(y, phys[i]) for i in range(PHYS_SEEDS)]))
    print(f"physics OOF (smoothed, 10f-CV, 3 seeds): {s1:.4f}")
    print(f"target std (mean-predictor RMSE): {y.std():.2f}\n")

    print(f"{'model':<28}{'features':<14}{'POOLED':>9}{'sd':>7}{'rho':>8}{'w*':>7}"
          f"{'blend*':>9}{'gain':>8}")
    rows = []
    for mname, mf in MODELS.items():
        for fname, (cols, fn) in FEATS.items():
            try:
                mats = [oof(tr, cols, fn, mf, s) for s in SEEDS]
            except Exception as exc:                      # a learner may not fit; say so
                print(f"{mname:<28}{fname:<14}  FAILED: {type(exc).__name__}")
                continue
            pooled = [rmse(y, m) for m in mats]
            rho = float(np.mean([np.corrcoef(phys[i] - y, mats[i] - y)[0, 1]
                                 for i in range(PHYS_SEEDS)]))
            s2 = float(np.mean(pooled))
            w, blend = optimal_blend(s1, s2, rho)
            rows.append({"model": mname, "features": fname, "pooled": s2,
                         "sd": float(np.std(pooled)), "rho": rho, "w_star": w,
                         "blend": blend, "gain": s1 - blend})
            print(f"{mname:<28}{fname:<14}{s2:>9.3f}{np.std(pooled):>7.3f}{rho:>8.3f}"
                  f"{w:>7.2f}{blend:>9.3f}{s1-blend:>+8.3f}")

    rows.sort(key=lambda r: r["blend"])
    print("\nRANKED BY PROJECTED BLEND (this is the metric that matters):")
    for r in rows[:6]:
        print(f"  {r['blend']:.3f}  gain {r['gain']:+.3f}   {r['model']} / {r['features']}"
              f"   (standalone {r['pooled']:.2f}, rho {r['rho']:.3f})")
    base = next(r for r in rows if r["model"].startswith("ExtraTrees leaf=1")
                and r["features"] == "shipped 13")
    print(f"\n  shipped baseline: blend {base['blend']:.3f} "
          f"(tree {base['pooled']:.2f}, rho {base['rho']:.3f})")
    print(f"  best beats it by {base['blend'] - rows[0]['blend']:+.3f} in projection")
    print("\n  Projection is ensemble ALGEBRA at the optimal GLOBAL weight. The shipped policy")
    print("  is regime-aware with a cutoff and an a-priori sigma, so anything promising here")
    print("  must go through joint_policy.py and be judged on LOSO before adoption.")

    (ARTIFACTS / "learner_eval.json").write_text(json.dumps(
        {"physics_oof": s1, "seeds": list(SEEDS), "rows": rows}, indent=2, default=float))
    print(f"\nwrote {ARTIFACTS / 'learner_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
