"""Reconcile the blend's gain band by band, in squared error, so the arithmetic closes.

The deck asserted three numbers that are each individually correct and do not visibly
reconcile: the entire gain comes from 13 rows, the blend LOSES on the 56 near-zero rows,
and the net over 150 rows is +0.27 RMSE. Working backwards from the RMSE levels, the 13
rows have to deliver the whole net improvement PLUS the near-zero loss, which implies a
mean squared-error improvement far larger than a mean need of +4.34 would suggest.

RMSE is not additive; squared error is. So this script does the accounting in SSE, where
the band contributions must sum to the total by construction, and then shows the error
DISTRIBUTION on the 13 rows -- because if heavy right skew is what reconciles it, the mean
alone cannot show that.

Basis note, which matters. Two defensible bases disagree slightly:
  * POOLED  -- all 3 seeds x 150 rows = 450 predictions treated as one sample. SSE is
    exactly additive across bands and the pooled RMSE is sqrt(SSE/450). This is the basis
    the reconciliation table uses, because it is the one that closes exactly.
  * PER-SEED -- RMSE computed per seed then averaged. This is the protocol the headline
    5.9352 -> 5.6541 is quoted under. Averaging RMSEs is not the same as pooling SSE, so
    the two differ in the third decimal. Both are reported; neither is wrong; they answer
    slightly different questions.
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
BANDS = [("dead   p<=0.5", -np.inf, 0.5),
         ("low    0.5-10", 0.5, 10.0),
         ("mid    10-60 ", 10.0, 60.0),
         ("above  p>60  ", 60.0, np.inf)]


def main() -> int:
    tr = load_train()
    y = tr[TARGET].to_numpy(dtype=float)
    z = np.load(ARTIFACTS / "oof_by_sigma.npz")
    phys_s = z[f"s_{NOISE_SIGMA_K}"]                       # (3, 150)
    tree_s = np.load(ARTIFACTS / "tree_oof.npy")
    w = json.loads((ARTIFACTS / "blend.json").read_text())["weight"]
    n_seeds = phys_s.shape[0]

    blend_s = np.vstack([apply_blend(phys_s[i], tree_s[i], w, BLEND_CUTOFF)
                         for i in range(n_seeds)])

    # ---- the two bases, stated side by side -------------------------------------
    per_seed_pre = [rmse(y, phys_s[i]) for i in range(n_seeds)]
    per_seed_post = [rmse(y, blend_s[i]) for i in range(n_seeds)]
    P = np.tile(y, n_seeds)
    pre_all, post_all = phys_s.ravel(), blend_s.ravel()
    sse_pre, sse_post = float(((pre_all - P) ** 2).sum()), float(((post_all - P) ** 2).sum())
    N = P.size
    pooled_pre, pooled_post = np.sqrt(sse_pre / N), np.sqrt(sse_post / N)

    print("BASIS COMPARISON -- both correct, they answer different questions")
    print(f"  per-seed RMSE then averaged : {np.mean(per_seed_pre):.4f} -> "
          f"{np.mean(per_seed_post):.4f}   gain {np.mean(per_seed_pre)-np.mean(per_seed_post):+.4f}"
          f"   <- the quoted headline")
    print(f"  pooled over {n_seeds} seeds ({N} preds): {pooled_pre:.4f} -> {pooled_post:.4f}"
          f"   gain {pooled_pre-pooled_post:+.4f}   <- the basis that closes exactly")
    print(f"  total SSE {sse_pre:.1f} -> {sse_post:.1f}   improvement {sse_pre-sse_post:.1f}")

    # ---- band-by-band SSE, which MUST sum ---------------------------------------
    print(f"\nRECONCILED ACCOUNTING (pooled, {N} predictions). SSE contributions sum to the total.")
    print(f"{'band':<15}{'n':>5}{'RMSE pre':>10}{'RMSE post':>11}{'dRMSE':>9}"
          f"{'SSE pre':>11}{'SSE post':>11}{'dSSE':>10}{'% of gain':>11}")
    band_ids = np.digitize(pre_all, [0.5, 10.0, 60.0])
    total_d = sse_pre - sse_post
    rows, check = [], 0.0
    for bi, (name, lo, hi) in enumerate(BANDS):
        m = band_ids == bi
        n = int(m.sum())
        sp = float(((pre_all[m] - P[m]) ** 2).sum())
        sq = float(((post_all[m] - P[m]) ** 2).sum())
        d = sp - sq
        check += d
        rp = np.sqrt(sp / n) if n else 0.0
        rq = np.sqrt(sq / n) if n else 0.0
        print(f"{name:<15}{n:>5}{rp:>10.3f}{rq:>11.3f}{rp-rq:>+9.3f}"
              f"{sp:>11.1f}{sq:>11.1f}{d:>+10.1f}{100*d/total_d:>10.1f}%")
        rows.append({"band": name.split()[0], "n": n, "rmse_pre": rp, "rmse_post": rq,
                     "rmse_delta": rp - rq, "sse_pre": sp, "sse_post": sq, "sse_delta": d,
                     "pct_of_gain": 100 * d / total_d})
    print(f"{'TOTAL':<15}{N:>5}{pooled_pre:>10.3f}{pooled_post:>11.3f}"
          f"{pooled_pre-pooled_post:>+9.3f}{sse_pre:>11.1f}{sse_post:>11.1f}{total_d:>+10.1f}"
          f"{100.0:>10.1f}%")
    print(f"  closure check: band dSSE sums to {check:.4f} vs total {total_d:.4f}  "
          f"(residual {abs(check-total_d):.2e})")

    # ---- the distribution on the gain-carrying band ------------------------------
    low = (pre_all > 0.5) & (pre_all <= 10.0)
    e_pre, e_post = P[low] - pre_all[low], P[low] - post_all[low]
    contrib = (P[low] - pre_all[low]) ** 2 - (P[low] - post_all[low]) ** 2
    order = np.argsort(-contrib)
    print(f"\nWHY THE MEAN MISLEADS -- error distribution on the {int(low.sum())} rows of the "
          f"0.5 < p <= 10 band")
    print(f"  need (y - phys):  mean {e_pre.mean():+.2f}   median {np.median(e_pre):+.2f}   "
          f"min {e_pre.min():+.2f}   max {e_pre.max():+.2f}")
    print(f"  |need| quantiles: p50 {np.percentile(np.abs(e_pre),50):.2f}  "
          f"p75 {np.percentile(np.abs(e_pre),75):.2f}  p90 {np.percentile(np.abs(e_pre),90):.2f}  "
          f"max {np.abs(e_pre).max():.2f}")
    top3 = contrib[order[:3]].sum()
    print(f"  squared-error improvement is tail-dominated: the top 3 of {int(low.sum())} rows "
          f"supply {top3:.0f} of {contrib.sum():.0f} ({100*top3/contrib.sum():.0f}%)")
    print(f"\n  {'y':>8}{'phys':>9}{'tree':>9}{'blended':>9}{'err pre':>9}{'err post':>10}{'dSSE':>10}")
    for i in order:
        print(f"  {P[low][i]:>8.2f}{pre_all[low][i]:>9.3f}{np.tile(tree_s.ravel(),1)[0]*0 + tree_s.ravel()[low][i]:>9.2f}"
              f"{post_all[low][i]:>9.3f}{e_pre[i]:>+9.2f}{e_post[i]:>+10.2f}{contrib[i]:>+10.1f}")

    out = {"basis": {"per_seed_mean_pre": float(np.mean(per_seed_pre)),
                     "per_seed_mean_post": float(np.mean(per_seed_post)),
                     "pooled_pre": float(pooled_pre), "pooled_post": float(pooled_post),
                     "n_predictions": int(N), "n_seeds": int(n_seeds)},
           "total_sse_pre": sse_pre, "total_sse_post": sse_post, "total_sse_delta": total_d,
           "bands": rows, "closure_residual": float(abs(check - total_d)),
           "low_band_distribution": {
               "n": int(low.sum()), "mean_need": float(e_pre.mean()),
               "median_need": float(np.median(e_pre)),
               "max_abs_need": float(np.abs(e_pre).max()),
               "top3_share_of_band_gain": float(top3 / contrib.sum()),
               "needs": [float(v) for v in np.sort(e_pre)]}}
    (ARTIFACTS / "blend_accounting.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ARTIFACTS / 'blend_accounting.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
