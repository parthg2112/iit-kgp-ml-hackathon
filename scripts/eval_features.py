"""Do reaction-engineering features make the tree a good enough ensemble member?

The tree is blended into the submission at 13% weight. Feature engineering cannot touch the
ODE -- it has physical parameters, not features -- so this is the only place features can
move our score. The question is not "is the tree better" but "is the BLEND better", and
those come apart:

Two-model ensemble algebra, which reproduces our shipped result exactly (physics 6.36, tree
16.37, error correlation 0.070 -> optimal w 0.887, predicted gain +0.30, measured +0.28):

    w* = (s2^2 - rho*s1*s2) / (s1^2 + s2^2 - 2*rho*s1*s2)

The gain depends on rho at least as much as on the tree's accuracy. A tree at RMSE 10 buys
+0.83 at rho=0.07 but only +0.08 at rho=0.5, because a better tree starts failing on the
same cliff rows the ODE fails on. So rho is reported as a first-class result.

REPORTING. Everything is POOLED out-of-fold RMSE over >=5 seeds -- one RMSE over all
predictions, which is what the competition computes over its 50 rows. Mean-of-fold-RMSEs
reads ~0.8 lower on this data and median ~1.9 lower, because sqrt is concave and the folds
are wildly heterogeneous (our per-fold spread runs 2.9 to 9.8). Those bases are printed too,
purely so the difference is visible.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import make_model
from src.data import FEATURE_COLUMNS, ROOT, TARGET, add_physics_features, load_train, rmse
from src.features import TRIAL_KINETICS, add_reaction_features, reaction_feature_columns
from src.physics import NOISE_SIGMA_K

ARTIFACTS = ROOT / "artifacts"
SEEDS = (0, 1, 2, 3, 4)
PHYSICS_SEEDS = (0, 1, 2)          # the seeds cached in oof_by_sigma.npz


def tree_oof(df, cols, feat_fn, seed):
    """Out-of-fold tree predictions. Features are a fixed deterministic transform of the
    inputs -- no fitting, no selection -- so building them outside the fold cannot leak.
    Anything fitted would have to move inside the loop."""
    X = feat_fn(df)[cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    oof = np.zeros(len(df))
    for tr_idx, te_idx in KFold(10, shuffle=True, random_state=seed).split(X):
        m = make_model(seed)
        m.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = np.clip(m.predict(X[te_idx]), 0.0, 100.0)
    return oof


def optimal_blend(s1, s2, rho):
    """Variance-optimal weight on model 1, and the resulting RMSE."""
    c = rho * s1 * s2
    denom = s1 ** 2 + s2 ** 2 - 2 * c
    w = 1.0 if denom <= 0 else float(np.clip((s2 ** 2 - c) / denom, 0.0, 1.0))
    v = w * w * s1 ** 2 + (1 - w) ** 2 * s2 ** 2 + 2 * w * (1 - w) * c
    return w, float(np.sqrt(v))


def main() -> int:
    tr = load_train()
    y = tr[TARGET].to_numpy(dtype=float)

    z = np.load(ARTIFACTS / "oof_by_sigma.npz")
    phys = z[f"s_{NOISE_SIGMA_K}"]                       # (3, 150) smoothed physics OOF
    s_phys = float(np.mean([rmse(y, phys[i]) for i in range(len(PHYSICS_SEEDS))]))

    one = TRIAL_KINETICS[:1]
    two = TRIAL_KINETICS[:2]
    f1 = lambda d: add_reaction_features(d, one)          # noqa: E731
    RAW = list(FEATURE_COLUMNS[:5])

    # ADDING the reaction block made the tree monotonically worse. The suspect is dilution,
    # not the features: ExtraTrees samples max_features=0.6 of the columns at every split,
    # so 26 extra columns crowd out the informative ones, and min_samples_leaf=1 lets it
    # memorize whatever noise survives. If that diagnosis is right, REPLACING the feature
    # set with a small physically-meaningful one should beat both.
    SETS = {
        "base (shipped, 13)":    (FEATURE_COLUMNS, add_physics_features),
        "+reaction x1 (21)":     (reaction_feature_columns(one), f1),
        "+reaction x2 (27)":     (reaction_feature_columns(two),
                                  lambda d: add_reaction_features(d, two)),
        "+reaction x4 (39)":     (reaction_feature_columns(TRIAL_KINETICS),
                                  add_reaction_features),
        # --- replacement sets: small, every column a named physical group ---
        "Yiso only (1)":         (["Yiso_0"], f1),
        "Yiso+tau_rel (2)":      (["Yiso_0", "logtau_rel_0"], f1),
        "minimal physics (5)":   (["Yiso_0", "logtau_rel_0", "ln_sel_0", "T_avg", "log_tau"],
                                  f1),
        "raw5+Yiso+tau_rel (7)": (RAW + ["Yiso_0", "logtau_rel_0"], f1),
        "raw5 only (5)":         (RAW, add_physics_features),
    }

    print(f"physics OOF (smoothed sigma={NOISE_SIGMA_K}, 10f-CV, {len(PHYSICS_SEEDS)} seeds):"
          f" {s_phys:.4f}\n")
    print(f"{'feature set':<24}{'n_feat':>7}{'POOLED':>9}{'per-seed sd':>13}"
          f"{'mean-of-fold':>14}{'median-fold':>13}{'worst fold':>12}")

    results = {}
    for name, (cols, fn) in SETS.items():
        pooled, meanf, medf, worst, oofs = [], [], [], [], []
        for s in SEEDS:
            oof = tree_oof(tr, cols, fn, s)
            oofs.append(oof)
            pooled.append(rmse(y, oof))
            fr = [rmse(y[b], oof[b])
                  for _, b in KFold(10, shuffle=True, random_state=s).split(oof)]
            meanf.append(np.mean(fr)); medf.append(np.median(fr)); worst.append(np.max(fr))
        results[name] = {"pooled": float(np.mean(pooled)),
                         "pooled_sd": float(np.std(pooled)),
                         "mean_of_folds": float(np.mean(meanf)),
                         "median_of_folds": float(np.mean(medf)),
                         "worst_fold": float(np.mean(worst)),
                         "n_features": len(cols),
                         "oof": np.vstack(oofs)}
        r = results[name]
        print(f"{name:<24}{r['n_features']:>7}{r['pooled']:>9.4f}{r['pooled_sd']:>13.4f}"
              f"{r['mean_of_folds']:>14.4f}{r['median_of_folds']:>13.4f}"
              f"{r['worst_fold']:>12.4f}")

    # ---- what actually decides it: correlation with the physics errors -------------
    print(f"\nDOES IT HELP THE BLEND? (rho = corr(physics error, tree error), "
          f"seeds {PHYSICS_SEEDS})")
    print(f"{'feature set':<24}{'tree':>8}{'rho':>8}{'w*':>7}{'blend*':>9}"
          f"{'vs physics':>12}")
    summary = {}
    for name, r in results.items():
        rhos = []
        for i, s in enumerate(PHYSICS_SEEDS):
            e_p, e_t = phys[i] - y, r["oof"][SEEDS.index(s)] - y
            rhos.append(np.corrcoef(e_p, e_t)[0, 1])
        rho = float(np.mean(rhos))
        w, blend = optimal_blend(s_phys, r["pooled"], rho)
        summary[name] = {k: v for k, v in r.items() if k != "oof"}
        summary[name].update({"rho": rho, "w_star": w, "blend_rmse": blend,
                              "gain_vs_physics": s_phys - blend})
        print(f"{name:<24}{r['pooled']:>8.3f}{rho:>8.3f}{w:>7.2f}{blend:>9.3f}"
              f"{s_phys - blend:>+12.3f}")

    best = min(summary, key=lambda k: summary[k]["blend_rmse"])
    base = summary["base (shipped)"]
    print(f"\nbest by projected blend: {best}")
    print(f"  blend {summary[best]['blend_rmse']:.3f} vs {base['blend_rmse']:.3f} for base"
          f"   -> {base['blend_rmse'] - summary[best]['blend_rmse']:+.3f}")
    print(f"  tree  {summary[best]['pooled']:.3f} vs {base['pooled']:.3f}"
          f"   rho {summary[best]['rho']:.3f} vs {base['rho']:.3f}")
    if summary[best]["rho"] > base["rho"] + 0.15:
        print("  NOTE: rho rose materially -- the better tree is failing on the same rows as")
        print("  the ODE, which is exactly what eats the projected gain.")
    print("\n  This projection is ensemble ALGEBRA, not a measured blend. It assumes the")
    print("  optimal global weight; the shipped policy is regime-aware with a cutoff and a")
    print("  jointly-derived sigma. If the gain here is real, joint_policy.py must re-derive")
    print("  (sigma, w, cutoff) together and LOSO is the number that decides adoption.")

    np.save(ARTIFACTS / "tree_oof_reaction.npy", results[best]["oof"][:len(PHYSICS_SEEDS)])
    (ARTIFACTS / "feature_eval.json").write_text(json.dumps(
        {"physics_oof": s_phys, "seeds": list(SEEDS), "best": best,
         "sets": summary}, indent=2, default=float))
    print(f"\nwrote {ARTIFACTS / 'feature_eval.json'}")
    print(f"wrote {ARTIFACTS / 'tree_oof_reaction.npy'}  (best set, "
          f"{len(PHYSICS_SEEDS)} seeds, aligned with the cached physics OOF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
