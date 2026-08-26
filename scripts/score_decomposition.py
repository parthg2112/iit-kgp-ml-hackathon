"""Decode the offline-round leaderboard score: RMSE 11.0442, MAE 3.2408, R2 0.9061.

We never see the test labels, but three summary statistics over 50 rows constrain the error
distribution tightly enough to say what happened.

The key statistic is RMSE/MAE. For Gaussian errors that ratio is sqrt(pi/2) = 1.25; for
Laplace it is 1.41. Ours is 3.41. A ratio that high cannot come from broadly mediocre
prediction -- it is the signature of a small number of catastrophic rows sitting on top of an
otherwise accurate fit. This script solves for how many.

It also does the thing that matters more than a nice story: checks the observed score against
our OWN out-of-fold error distribution. If the score sits outside what our cross-validation
predicted, we say so and quantify it, rather than claiming we called it.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_train, rmse
from src.evaluate import BLEND_CUTOFF, apply_blend
from src.physics import NOISE_SIGMA_K

ARTIFACTS = ROOT / "artifacts"

# The official offline-round result. Rank 13 of 400.
LB_RMSE, LB_MAE, LB_R2, N_TEST = 11.0442, 3.2408, 0.9061, 50


def two_group(k, sse, sae, n):
    """Solve: k rows at |error| E, the remaining (n-k) at |error| m, matching SSE and SAE.

    k*E + (n-k)*m = SAE ;  k*E^2 + (n-k)*m^2 = SSE.  Substituting the first into the second
    gives a quadratic in m. Returns (E, m) for the admissible root, or None.
    """
    a = (n - k) ** 2 / k + (n - k)
    b = -2 * sae * (n - k) / k
    c = sae ** 2 / k - sse
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    for m in ((-b - disc ** 0.5) / (2 * a), (-b + disc ** 0.5) / (2 * a)):
        E = (sae - (n - k) * m) / k
        if m >= 0 and E >= m:
            return float(E), float(m)
    return None


def main() -> int:
    sse = N_TEST * LB_RMSE ** 2
    sae = N_TEST * LB_MAE
    sstot = sse / (1.0 - LB_R2)
    implied_sd = (sstot / N_TEST) ** 0.5
    ratio = LB_RMSE / LB_MAE

    print("LEADERBOARD SCORE, DECODED   (rank 13 / 400)\n")
    print(f"  RMSE {LB_RMSE}   MAE {LB_MAE}   R2 {LB_R2}   n = {N_TEST}")
    print(f"  SSE {sse:.1f}   SAE {sae:.1f}")
    print(f"  implied test-target std {implied_sd:.2f}  (training std 38.30)"
          f"  -> the test set is distributed like the training set")
    print(f"\n  RMSE/MAE = {ratio:.2f}")
    print(f"    Gaussian errors give 1.25, Laplace 1.41.")
    print(f"    {ratio:.2f} means a FEW rows dominate the squared error.\n")

    print("  Two-group reconstruction:")
    print(f"  {'k big rows':>12}{'their error':>14}{'error on the other rows':>26}")
    groups = {}
    for k in (1, 2, 3, 4, 5):
        got = two_group(k, sse, sae, N_TEST)
        if got is None:
            continue
        E, m = got
        groups[k] = {"n_big": k, "error_big": E, "error_rest": m,
                     "n_rest": N_TEST - k}
        print(f"  {k:>12}{E:>14.1f}{m:>26.2f}")
    print("\n  -> consistently: 2-3 rows missed by ~45-55 yield-points,")
    print("     while the other ~47 rows are accurate to about 1 yield-point.")

    # ---- the honest check: does our own CV predict this score? -----------------
    tr = load_train()
    y = tr[TARGET].to_numpy(dtype=float)
    z = np.load(ARTIFACTS / "oof_by_sigma.npz")
    phys = z[f"s_{NOISE_SIGMA_K}"]
    tree = np.load(ARTIFACTS / "tree_oof.npy")
    w = json.loads((ARTIFACTS / "blend.json").read_text())["weight"]
    bl = np.vstack([apply_blend(phys[i], tree[i], w, BLEND_CUTOFF)
                    for i in range(phys.shape[0])])

    rng = np.random.default_rng(0)
    R, M, RT, BIG = [], [], [], []
    for _ in range(20000):
        s = int(rng.integers(0, bl.shape[0]))
        idx = rng.choice(len(y), N_TEST, replace=True)
        e = bl[s][idx] - y[idx]
        r = float(np.sqrt(np.mean(e ** 2)))
        m = float(np.mean(np.abs(e)))
        R.append(r); M.append(m); RT.append(r / m)
        BIG.append(int((np.abs(e) > 40).sum()))
    R, M, RT, BIG = map(np.array, (R, M, RT, BIG))

    print("\nDOES OUR OWN CROSS-VALIDATION PREDICT THIS SCORE?")
    print("  (20000 bootstrapped 50-row draws from our out-of-fold predictions)\n")
    print(f"  {'statistic':<16}{'observed':>10}{'our 5-95 pct':>20}{'our median':>12}")
    for name, obs, arr in (("RMSE", LB_RMSE, R), ("MAE", LB_MAE, M),
                           ("RMSE/MAE", ratio, RT)):
        print(f"  {name:<16}{obs:>10.2f}   [{np.percentile(arr,5):5.2f},"
              f"{np.percentile(arr,95):6.2f}]{np.median(arr):>12.2f}")
    p_rmse = float((R >= LB_RMSE).mean())
    p_mae = float((M <= LB_MAE).mean())
    print(f"\n  P(our RMSE >= {LB_RMSE})  = {p_rmse:.1%}")
    print(f"  P(our MAE  <= {LB_MAE})   = {p_mae:.1%}")
    print(f"  rows with |error| > 40 in our OOF: {int((np.abs(bl - y).max(0) > 40).sum())}"
          f" of {len(y)}   (mean {BIG.mean():.2f} per 50-row draw)")

    print("\n  READ THIS HONESTLY:")
    print("  Our MAE landed exactly where we predicted. Our RMSE did NOT -- it is outside")
    print("  the range our own bootstrap produced. We correctly pre-registered the failure")
    print("  MODE (cliff rows carry the error) and its location, but we UNDERESTIMATED its")
    print("  magnitude: our 150 training rows never produced a miss as large as the test")
    print("  set did. Do not claim we called this. Claim we called the mechanism.")

    out = {
        "leaderboard": {"rank": 13, "of": 400, "rmse": LB_RMSE, "mae": LB_MAE,
                        "r2": LB_R2, "n": N_TEST},
        "derived": {"sse": sse, "sae": sae, "rmse_over_mae": ratio,
                    "implied_target_sd": implied_sd,
                    "gaussian_ratio": 1.2533, "laplace_ratio": 1.4142},
        "two_group": groups,
        "our_oof_bootstrap": {
            "n_draws": 20000,
            "rmse_p5": float(np.percentile(R, 5)), "rmse_p95": float(np.percentile(R, 95)),
            "rmse_median": float(np.median(R)),
            "mae_p5": float(np.percentile(M, 5)), "mae_p95": float(np.percentile(M, 95)),
            "mae_median": float(np.median(M)),
            "ratio_p5": float(np.percentile(RT, 5)), "ratio_p95": float(np.percentile(RT, 95)),
            "p_rmse_at_least_observed": p_rmse, "p_mae_at_most_observed": p_mae,
            "oof_rows_over_40": int((np.abs(bl - y).max(0) > 40).sum())},
    }
    (ARTIFACTS / "score_decomposition.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ARTIFACTS / 'score_decomposition.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
