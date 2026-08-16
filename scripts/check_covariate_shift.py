"""Is the test set drawn from the same distribution as the training set?

This matters for model choice, not just curiosity. If test inputs sit inside the training
envelope, a flexible model is interpolating and its CV estimate is roughly honest. If they
sit outside it, every extra degree of freedom is extrapolating into territory the fit never
saw, and CV will flatter it.

Three checks, weakest to strongest:
  1. RANGE   -- does any test row fall outside the per-feature training min/max?
  2. KS      -- are the marginal distributions distinguishable?
  3. AUC     -- can a classifier tell a train row from a test row at all? This is the real
                test, because it sees the JOINT distribution; the marginals can each match
                while the combination is still shifted. AUC ~ 0.5 means no detectable shift.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import RAW_FEATURES, ROOT, add_physics_features, load_test, load_train

DERIVED = ["tau", "log_tau", "T_avg", "delta_T", "inv_T_avg"]


def main() -> int:
    tr, te = load_train(), load_test()
    ftr, fte = add_physics_features(tr), add_physics_features(te)

    print("1. RANGE CHECK -- does the test set stay inside the training envelope?\n")
    print(f"{'feature':<24}{'train min':>11}{'train max':>11}{'test min':>11}{'test max':>11}"
          f"{'outside':>9}")
    n_out_total = 0
    for c in RAW_FEATURES:
        a, b = tr[c].to_numpy(float), te[c].to_numpy(float)
        out = int(((b < a.min()) | (b > a.max())).sum())
        n_out_total += out
        flag = f"  {out} rows" if out else "  -"
        print(f"{c:<24}{a.min():>11.3f}{a.max():>11.3f}{b.min():>11.3f}{b.max():>11.3f}{flag:>9}")
    print(f"\n  {n_out_total} test values fall outside the training range "
          f"(of {len(te)*len(RAW_FEATURES)} checked)")

    print("\n2. KS TEST on marginals (p < 0.05 would indicate a shifted marginal)\n")
    print(f"{'feature':<24}{'KS stat':>10}{'p-value':>10}")
    for c in RAW_FEATURES + DERIVED:
        s = ks_2samp(ftr[c].to_numpy(float), fte[c].to_numpy(float))
        mark = "  <- shifted" if s.pvalue < 0.05 else ""
        print(f"{c:<24}{s.statistic:>10.3f}{s.pvalue:>10.3f}{mark}")

    print("\n3. TWO-SAMPLE CLASSIFIER -- can anything tell train from test?\n")
    cols = RAW_FEATURES + DERIVED
    X = np.vstack([ftr[cols].to_numpy(float), fte[cols].to_numpy(float)])
    lab = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=0)
    aucs = []
    for seed in (0, 1, 2, 3, 4):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(clf, X, lab, cv=cv, method="predict_proba")[:, 1]
        # AUC without sklearn.metrics import noise: rank-based Mann-Whitney form
        order = np.argsort(p)
        ranks = np.empty(len(p), float)
        ranks[order] = np.arange(1, len(p) + 1)
        n1, n0 = lab.sum(), (1 - lab).sum()
        aucs.append((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    auc = float(np.mean(aucs))
    print(f"  cross-validated AUC over 5 seeds: {auc:.3f} +/- {np.std(aucs):.3f}")
    print(f"  (0.50 = indistinguishable, 1.00 = perfectly separable)")

    verdict = ("NO detectable covariate shift -- the test set is an interpolation problem"
               if auc < 0.60 else
               "SHIFT DETECTED -- test rows are distinguishable; flexible models will extrapolate")
    print(f"\n  VERDICT: {verdict}")

    print("\n  What this does and does not license:")
    print("  - It says the test INPUTS look like the training inputs, so CV is a fair proxy.")
    print("  - It does NOT say a complex model is safe. With n=150 the binding constraint is")
    print("    sample size, not distribution shift: a model with more effective parameters")
    print("    than the data supports overfits inside the envelope just as happily.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
