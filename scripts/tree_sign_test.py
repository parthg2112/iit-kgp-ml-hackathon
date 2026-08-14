"""Where does the tree's +0.27 actually come from, and is it bias or variance?

The first sign test was run on the MID-RANGE stratum (10 < p <= 60), which contributes
NONE of the gain -- it is the band where the tree is wrong-signed. That was the wrong
stratum to test. The entire gain lives in the 13 rows with 0.5 < p <= 10, where the tree
supplies +3.29 against +4.34 needed.

This re-runs the sign test restricted to the band that carries the gain, and reports every
band so the stratum a number came from is never ambiguous again.

  >= 11/13 aligned -> band-localized BIAS CORRECTION, and the
     "variance reduction, never bias correction" line in the docs is wrong.
  <  11/13         -> variance reduction stands, now tested where it matters.

Also reports, independently of the tree: the physics under-predicts by ~4 points in the
(0.5, 10] band. That is a locality where the ODE is off and belongs in the deck either way.

Finally, test-set coverage: the blend region's helpful:harmful composition on the test set
must resemble the training composition, or the measured gain was measured on a different
population than the one we face.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import fit_predict
from src.data import ROOT, TARGET, load_test, load_train, rmse
from src.evaluate import BLEND_CUTOFF, apply_blend
from src.physics import NOISE_SIGMA_K, SUBMIT_STEPS, predict_smoothed

ARTIFACTS = ROOT / "artifacts"
BANDS = [("dead   p<=0.5", -np.inf, 0.5),
         ("low    0.5-10", 0.5, 10.0),
         ("mid    10-60 ", 10.0, 60.0)]


def main() -> int:
    tr = load_train()
    y = tr[TARGET].to_numpy(dtype=float)

    z = np.load(ARTIFACTS / "oof_by_sigma.npz")
    phys = z[f"s_{NOISE_SIGMA_K}"].mean(0)          # smoothed physics OOF, averaged over seeds
    tree = np.load(ARTIFACTS / "tree_oof.npy").mean(0)
    w = json.loads((ARTIFACTS / "blend.json").read_text())["weight"]

    blended = apply_blend(phys, tree, w, BLEND_CUTOFF)
    print(f"policy: sigma={NOISE_SIGMA_K} w={w:.3f} cutoff={BLEND_CUTOFF:.0f}")
    print(f"OOF RMSE (seed-averaged predictions, then scored) "
          f"{rmse(y, phys):.4f} -> {rmse(y, blended):.4f}")
    # Per-seed is the accounting the docs and LOSO use; averaging predictions across seeds
    # first is itself a variance reduction and reads lower. Quote per-seed.
    ps = [rmse(y, z[f"s_{NOISE_SIGMA_K}"][i]) for i in range(3)]
    bs = [rmse(y, apply_blend(z[f"s_{NOISE_SIGMA_K}"][i],
                              np.load(ARTIFACTS / "tree_oof.npy")[i], w, BLEND_CUTOFF))
          for i in range(3)]
    print(f"OOF RMSE (per-seed, then averaged -- the quoted protocol) "
          f"{np.mean(ps):.4f} -> {np.mean(bs):.4f}  gain {np.mean(ps) - np.mean(bs):+.4f}\n")

    pull = tree - phys                 # direction the tree wants to move the prediction
    need = y - phys                    # direction that would reduce error
    applied = (1.0 - w) * pull

    print("SIGN TEST BY BAND -- stated explicitly, because stratum choice was the first test's flaw")
    print(f"{'band':<15}{'n':>4}{'aligned':>9}{'%':>7}{'p(2s)':>8}"
          f"{'mean pull':>11}{'applied':>9}{'needed':>9}{'gain':>8}")
    out = {}
    for name, lo, hi in BANDS:
        m = (phys > lo) & (phys <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        agree = int((np.sign(pull[m]) == np.sign(need[m])).sum())
        p = binomtest(agree, n, 0.5).pvalue
        gain = rmse(y[m], phys[m]) - rmse(y[m], blended[m])
        print(f"{name:<15}{n:>4}{agree:>6}/{n:<3}{100*agree/n:>6.0f}%{p:>8.3f}"
              f"{pull[m].mean():>11.2f}{applied[m].mean():>9.2f}{need[m].mean():>9.2f}{gain:>8.3f}")
        out[name.split()[0]] = {"n": n, "aligned": agree, "p": float(p),
                                "mean_pull": float(pull[m].mean()),
                                "mean_applied": float(applied[m].mean()),
                                "mean_needed": float(need[m].mean()),
                                "band_rmse_gain": float(gain)}

    low = (phys > 0.5) & (phys <= 10.0)
    n_low, a_low = int(low.sum()), int((np.sign(pull[low]) == np.sign(need[low])).sum())
    p_low = binomtest(a_low, n_low, 0.5).pvalue
    verdict = "BIAS CORRECTION (band-localized)" if a_low >= 11 else "VARIANCE REDUCTION"
    print(f"\nGATE on the gain-carrying band (0.5 < p <= 10): {a_low}/{n_low} aligned, "
          f"p={p_low:.4f}  ->  {verdict}")
    print(f"  (bar was >= 11/{n_low})")

    # Is the gate verdict an artifact of averaging the tree over seeds? Check each seed.
    tree_all = np.load(ARTIFACTS / "tree_oof.npy")
    per_seed = []
    for i in range(tree_all.shape[0]):
        p_i = z[f"s_{NOISE_SIGMA_K}"][i]
        m_i = (p_i > 0.5) & (p_i <= 10.0)
        a_i = int((np.sign(tree_all[i][m_i] - p_i[m_i]) == np.sign(y[m_i] - p_i[m_i])).sum())
        per_seed.append((int(m_i.sum()), a_i))
    print("  per-seed (not averaged): " +
          "  ".join(f"seed {i}: {a}/{n}" for i, (n, a) in enumerate(per_seed)))

    # --- Independent of the tree: is the ODE itself off in this band? ---------
    print(f"\nPHYSICS-ONLY finding in the same band (holds whether or not we keep the tree):")
    print(f"  mean signed residual y - phys = {need[low].mean():+.3f}  "
          f"(median {np.median(need[low]):+.3f}, {int((need[low] > 0).sum())}/{n_low} under-predicted)")
    print(f"  band RMSE {rmse(y[low], phys[low]):.3f} vs whole-set {rmse(y, phys):.3f}")
    print("  READ THIS CAREFULLY: mean +4.34 but median +1.05. This is NOT a uniform 4-point")
    print("  offset -- it is a few badly-missed rows, which is the 29%-of-rows/91%-of-error")
    print("  decomposition showing up again, not a separate systematic bias.")

    # --- Is the dead band's 30%/p=0.005 a finding, or forced by construction? A sign test
    #     only informs where `need` actually varies in sign and `pull` can go either way.
    dead = phys <= 0.5
    print("\nIS THE DEAD-BAND ANTI-ALIGNMENT MECHANICAL? (sign test needs both signs available)")
    print(f"  need = y - phys : {int((need[dead] > 0).sum())} positive / "
          f"{int((need[dead] <= 0).sum())} non-positive of {int(dead.sum())}")
    print(f"  pull = tree-phys: {int((pull[dead] > 0).sum())} positive / "
          f"{int((pull[dead] <= 0).sum())} non-positive  (a tree cannot predict below its floor)")
    print(f"  -> if pull is ~always positive the sign test is near-tautological here.")
    print(f"  The non-tautological statement is the RMSE cost of blending this band: "
          f"{rmse(y[dead], phys[dead]) - rmse(y[dead], blended[dead]):+.3f}")

    # --- The dead band is the largest stratum and the blend LOSES there. What would a
    #     lower gate buy? Reported, NOT adopted: it is a third tuned rule, measured
    #     in-sample here, on the same rows that would judge it.
    print("\nTEMPTING PATCH, NOT ADOPTED: gate the blend from below as well")
    for lo_gate in (0.0, 0.5, 1.0, 2.0, 5.0):
        v = []
        for i in range(3):
            p_i, t_i = z[f"s_{NOISE_SIGMA_K}"][i], np.load(ARTIFACTS / "tree_oof.npy")[i]
            b_i = apply_blend(p_i, t_i, w, BLEND_CUTOFF)
            b_i = np.where(p_i <= lo_gate, np.clip(p_i, 0, 100), b_i)
            v.append(rmse(y, b_i))
        tag = "  <- shipped (no lower gate)" if lo_gate == 0.0 else ""
        print(f"  blend only where p > {lo_gate:4.1f}: {np.mean(v):.4f}{tag}")
    print("  in-sample only; adopting would need leave-one-seed-out and a fourth free knob")

    # --- Test-set coverage: same composition inside the blend region? --------
    print("\nTEST-SET COVERAGE inside the blend region (p <= cutoff)")
    te = load_test()
    x = json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"]
    phys_te = predict_smoothed(x, te, NOISE_SIGMA_K, n_steps=SUBMIT_STEPS)
    tree_te = np.mean([fit_predict(tr, te, seed=s) for s in (0, 1, 2)], axis=0)

    tr_in = phys <= BLEND_CUTOFF
    te_in = phys_te <= BLEND_CUTOFF
    # Two defensible definitions of "helpful"; report both, since they disagree.
    h_sign = int((np.sign(pull[tr_in]) == np.sign(need[tr_in])).sum())
    h_err = int((np.abs(blended[tr_in] - y[tr_in]) < np.abs(phys[tr_in] - y[tr_in])).sum())
    n_in = int(tr_in.sum())
    print(f"  train rows in region: {n_in}/{len(y)}")
    print(f"    helpful by SIGN of pull      {h_sign}:{n_in - h_sign} = {h_sign / n_in:.2f}")
    print(f"    helpful by ERROR REDUCED     {h_err}:{n_in - h_err} = {h_err / n_in:.2f}")
    print(f"  test rows in region: {int(te_in.sum())}/{len(te)} "
          f"({te_in.mean():.0%} vs train {tr_in.mean():.0%})")
    print("  NOTE: test-side helpfulness is UNKNOWABLE (no labels). The only checkable")
    print("  coverage claim is that the BAND COMPOSITION matches -- shown below.")
    for name, lo, hi in BANDS:
        mt = (phys_te > lo) & (phys_te <= hi)
        mr = (phys > lo) & (phys <= hi)
        print(f"    {name}: test {int(mt.sum()):3d} ({mt.mean():5.1%})   "
              f"train {int(mr.sum()):3d} ({mr.mean():5.1%})")

    out["gate"] = {"band": "0.5 < p <= 10", "n": n_low, "aligned": a_low, "p": float(p_low),
                   "verdict": verdict, "bar": 11}
    out["physics_band_bias"] = {"mean_signed_residual": float(need[low].mean()),
                                "band_rmse": float(rmse(y[low], phys[low]))}
    out["coverage"] = {"train_in_region": int(tr_in.sum()), "test_in_region": int(te_in.sum()),
                       "train_frac": float(tr_in.mean()), "test_frac": float(te_in.mean()),
                       "train_helpful_by_sign": h_sign, "train_helpful_by_error": h_err,
                       "helpful_ratio_by_error": float(h_err / n_in)}
    (ARTIFACTS / "tree_sign_test.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ARTIFACTS / 'tree_sign_test.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
