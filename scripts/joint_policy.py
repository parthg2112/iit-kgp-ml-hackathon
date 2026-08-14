"""Joint re-derivation of the prediction policy: (sigma, weight, cutoff) together.

Three things this fixes relative to the first attempt:

1. JOINT, not sequential. Picking sigma first and then re-deriving the blend lands wherever
   the ordering happens to put you.
2. HONEST sigma accounting. Choosing sigma on the same CV that is then quoted makes the
   reported gain an argmax. The headline number here fixes sigma *a priori* at the
   errors-in-variables median (1.67 K), so CV only confirms an independently estimated
   quantity. The argmax is reported too, labelled as optimistic.
3. BOUNDARY SLICES measured separately. A global mean can improve while both ends regress:
   - top end: the model saturates at 100, so averaging near a predicted 99 can almost only
     push down. The cutoff rule exists precisely to stop the tree shrinking the top;
     smoothing can reintroduce that through a different door.
   - dead-regime edge: deep in the dead regime dY/dT = 0 so smoothing does nothing, but rows
     at the edge can lift off zero against a true 0.000.

Only the physics component is smoothed. The tree is fit on observed inputs and is already
implicitly averaged over them; blurring it too would double-count the noise.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_train, ode_inputs, rmse
from src.evaluate import apply_blend
from src.physics import DEFAULT_STEPS, LOWER, UPPER, integrate, predict_smoothed, residuals

ARTIFACTS = ROOT / "artifacts"
SIGMAS = [0.0, 1.0, 1.67, 2.0, 2.5, 3.0]
WEIGHTS = np.linspace(0.70, 1.00, 31)
CUTOFFS = [40.0, 50.0, 60.0, 70.0, 85.0, 101.0]
SEEDS = (0, 1, 2)
EIV_SIGMA = 1.67          # a-priori value from scripts/eiv_noise.py -- NOT tuned on CV


def build_oof(tr, y):
    """Fit folds once per seed, then evaluate every sigma on the same fits."""
    x0 = np.array(json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"])
    oof = {s: np.zeros((len(SEEDS), len(tr))) for s in SIGMAS}
    for si, seed in enumerate(SEEDS):
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(tr):
            sub = tr.iloc[tr_idx]
            r = least_squares(residuals, x0=x0, bounds=(LOWER, UPPER),
                              args=(ode_inputs(sub), sub[TARGET].to_numpy(float),
                                    DEFAULT_STEPS, None),
                              x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
            held = tr.iloc[te_idx]
            for s in SIGMAS:
                oof[s][si, te_idx] = (integrate(r.x, ode_inputs(held), n_steps=DEFAULT_STEPS)
                                      if s == 0 else predict_smoothed(r.x, held, s))
        print(f"  seed {seed} folds fitted", flush=True)
    return oof


def best_triple(y, oof, tree, seed_idx, sigmas=SIGMAS):
    """Joint argmin over (sigma, w, cutoff) on the given seeds."""
    best = None
    for s in sigmas:
        p = oof[s][seed_idx].mean(0)
        t = tree[seed_idx].mean(0)
        for w in WEIGHTS:
            for c in CUTOFFS:
                v = rmse(y, apply_blend(p, t, w, c))
                if best is None or v < best[0]:
                    best = (v, s, float(w), float(c))
    return best


def main() -> int:
    tr = load_train()
    y = tr[TARGET].to_numpy(dtype=float)
    tree = np.load(ARTIFACTS / "tree_oof.npy")

    cache = ARTIFACTS / "oof_by_sigma.npz"
    if cache.exists():
        z = np.load(cache)
        oof = {float(k.split("_")[1]): z[k] for k in z.files}
        print("loaded cached OOF by sigma")
    else:
        print("fitting folds (once) and evaluating every sigma on the same fits:")
        oof = build_oof(tr, y)
        np.savez(cache, **{f"s_{s}": v for s, v in oof.items()})

    print(f"\npure-physics OOF by sigma (mean over {len(SEEDS)} seeds):")
    for s in SIGMAS:
        print(f"  sigma {s:5.2f} K -> {np.mean([rmse(y, oof[s][i]) for i in range(len(SEEDS))]):.4f}")

    all_idx = list(range(len(SEEDS)))
    v_argmax, s_argmax, w_argmax, c_argmax = best_triple(y, oof, tree, all_idx)
    print(f"\njoint in-sample argmax: sigma={s_argmax:.2f} w={w_argmax:.3f} cutoff={c_argmax:.0f}"
          f"  -> {v_argmax:.4f}   [OPTIMISTIC: sigma chosen on this same CV]")

    v_apri, _, w_apri, c_apri = best_triple(y, oof, tree, all_idx, sigmas=[EIV_SIGMA])
    print(f"a-priori sigma={EIV_SIGMA} (from errors-in-variables, not tuned here): "
          f"w={w_apri:.3f} cutoff={c_apri:.0f} -> {v_apri:.4f}")

    # --- Held-out seed discipline on the whole triple -----------------------
    print("\nleave-one-seed-out on the FULL triple (chosen on 2 seeds, scored on the 3rd):")
    rows = []
    for i in all_idx:
        others = [j for j in all_idx if j != i]
        _, s_i, w_i, c_i = best_triple(y, oof, tree, others)
        _, _, wa_i, ca_i = best_triple(y, oof, tree, others, sigmas=[EIV_SIGMA])
        held_joint = rmse(y, apply_blend(oof[s_i][i], tree[i], w_i, c_i))
        held_apri = rmse(y, apply_blend(oof[EIV_SIGMA][i], tree[i], wa_i, ca_i))
        # PREVIOUS policy (superseded by this script's own result): unsmoothed
        # physics, w=0.910, cutoff=60. Kept as the comparison baseline; the
        # shipped policy is (sigma, w, cutoff) = (1.67, 0.87, 60) in blend.json.
        held_base = rmse(y, apply_blend(oof[0.0][i], tree[i], 0.910, 60.0))
        pure_sm = rmse(y, oof[EIV_SIGMA][i])
        rows.append((held_base, held_joint, held_apri, pure_sm, s_i, w_i, c_i))
        print(f"  hold {i}: chosen (sigma={s_i:.2f}, w={w_i:.3f}, c={c_i:.0f}) -> {held_joint:.4f} | "
              f"a-priori sigma -> {held_apri:.4f} | current policy {held_base:.4f} | "
              f"pure smoothed {pure_sm:.4f}")
    R = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
    print(f"\n  mean: current {R[:,0].mean():.4f} | joint-argmax {R[:,1].mean():.4f} "
          f"| a-priori sigma {R[:,2].mean():.4f} | pure smoothed {R[:,3].mean():.4f}")
    print(f"  gain of a-priori-sigma policy over current: {R[:,0].mean()-R[:,2].mean():+.4f} "
          f"(all seeds better: {bool(np.all(R[:,2] < R[:,0]))})")

    # --- Does the tree still contribute anything once smoothing is in? ------
    print("\nis the tree still earning its weight? (a-priori sigma, cutoff fixed at 60)")
    for w in (0.0, 0.5, 0.8, 0.89, 0.95, 1.0):
        v = np.mean([rmse(y, apply_blend(oof[EIV_SIGMA][i], tree[i], w, 60.0)) for i in all_idx])
        tag = "  <- pure physics" if w == 1.0 else ("  <- tree only" if w == 0.0 else "")
        print(f"  w={w:4.2f} -> {v:.4f}{tag}")

    # --- Boundary slices ----------------------------------------------------
    print("\nBOUNDARY SLICE (a): top end -- does smoothing shrink high predictions?")
    for i in all_idx:
        p0, ps = oof[0.0][i], oof[EIV_SIGMA][i]
        m = p0 > 85
        print(f"  seed {i}: n={m.sum():3d}  RMSE raw {rmse(y[m], p0[m]):.3f} -> "
              f"smoothed {rmse(y[m], ps[m]):.3f}   max {p0.max():.2f} -> {ps.max():.2f}")

    print("\nBOUNDARY SLICE (b): dead-regime edge -- are true zeros lifted off zero?")
    dead = y < 1.0
    for i in all_idx:
        p0, ps = oof[0.0][i], oof[EIV_SIGMA][i]
        lifted = int(((p0[dead] < 0.5) & (ps[dead] > 1.0)).sum())
        print(f"  seed {i}: n={dead.sum():3d}  RMSE raw {rmse(y[dead], p0[dead]):.3f} -> "
              f"smoothed {rmse(y[dead], ps[dead]):.3f}   rows lifted 0->1+: {lifted}")

    (ARTIFACTS / "joint_policy.json").write_text(json.dumps({
        "eiv_sigma": EIV_SIGMA, "argmax": {"sigma": s_argmax, "w": w_argmax, "cutoff": c_argmax,
                                           "oof": v_argmax},
        "apriori": {"sigma": EIV_SIGMA, "w": w_apri, "cutoff": c_apri, "oof": v_apri},
        "loso_mean": {"current": float(R[:, 0].mean()), "joint": float(R[:, 1].mean()),
                      "apriori": float(R[:, 2].mean()), "pure_smoothed": float(R[:, 3].mean())},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
