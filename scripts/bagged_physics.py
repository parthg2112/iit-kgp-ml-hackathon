"""Can bagging the physics model replace the tree entirely?

The tree's contribution was measured as variance reduction from a decorrelated estimator,
NOT bias correction -- and it introduces ~+1.08 of upward bias to buy that variance. If the
gain is purely averaging, we may be able to get it from the physics model alone: refit on
bootstrap resamples and average. Same mechanism, one model instead of two, and no
systematic upward pull.

Evaluated out-of-fold: inside each CV fold, bootstrap the fold-TRAINING rows, refit, and
average the held-out predictions. Compared against the single-fit smoothed physics and
against the shipped physics+tree blend.
"""
import json, sys
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
sys.path.insert(0, '.')
from src.data import ROOT, TARGET, load_train, ode_inputs, rmse
from src.evaluate import apply_blend
from src.physics import DEFAULT_STEPS, LOWER, UPPER, NOISE_SIGMA_K, predict_smoothed, residuals

B = 12          # bootstrap replicates per fold
SEEDS = (0, 1, 2)

def _one(job):
    x0, sub_idx, held_idx, rep_seed = job
    tr = load_train()
    sub = tr.iloc[sub_idx]
    rng = np.random.default_rng(rep_seed)
    boot = sub.iloc[rng.integers(0, len(sub), len(sub))] if rep_seed >= 0 else sub
    r = least_squares(residuals, x0=x0, bounds=(LOWER, UPPER),
                      args=(ode_inputs(boot), boot[TARGET].to_numpy(float), DEFAULT_STEPS, None),
                      x_scale="jac", xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=250)
    return held_idx, predict_smoothed(r.x, tr.iloc[held_idx], NOISE_SIGMA_K)

if __name__ == "__main__":
    tr = load_train(); y = tr[TARGET].to_numpy(float)
    x0 = np.array(json.loads((ROOT/"artifacts"/"physics_params.json").read_text())["vector"])
    jobs = []
    for si, seed in enumerate(SEEDS):
        for fi, (a, b_) in enumerate(KFold(10, shuffle=True, random_state=seed).split(tr)):
            for rep in range(B):
                jobs.append((x0, a, b_, seed*1000 + fi*100 + rep))
    print(f"{len(jobs)} bootstrap fits on {min(10, cpu_count()-1)} workers", flush=True)
    acc = {s: np.zeros((B, len(tr))) for s in SEEDS}
    with Pool(min(10, max(1, cpu_count()-1))) as p:
        out = p.map(_one, jobs)
    k = 0
    for si, seed in enumerate(SEEDS):
        for fi in range(10):
            for rep in range(B):
                held, pred = out[k]; acc[seed][rep, held] = pred; k += 1
    bag = np.array([acc[s].mean(0) for s in SEEDS])
    np.save(ROOT/"artifacts"/"bagged_oof.npy", bag)
    single = np.load(ROOT/"artifacts"/"oof_by_sigma.npz")["s_1.67"]
    T = np.load(ROOT/"artifacts"/"tree_oof.npy")
    print(f"\n{'':28s} {'seed0':>8s}{'seed1':>8s}{'seed2':>8s}{'mean':>9s}")
    rows = {"single smoothed physics": [rmse(y, single[i]) for i in range(3)],
            "BAGGED smoothed physics": [rmse(y, bag[i]) for i in range(3)],
            "single + tree (shipped)": [rmse(y, apply_blend(single[i], T[i], 0.87, 60.0)) for i in range(3)],
            "bagged + tree":           [rmse(y, apply_blend(bag[i], T[i], 0.87, 60.0)) for i in range(3)]}
    for n, v in rows.items():
        print(f"{n:28s} " + "".join(f"{x:8.4f}" for x in v) + f"{np.mean(v):9.4f}")
    gb = np.mean(rows["single smoothed physics"]) - np.mean(rows["BAGGED smoothed physics"])
    gt = np.mean(rows["single smoothed physics"]) - np.mean(rows["single + tree (shipped)"])
    print(f"\nbagging gain  {gb:+.4f}   (tree gain {gt:+.4f})")
    print(f"mean signed residual: bagged {np.mean(bag.mean(0)-y):+.3f} vs "
          f"shipped blend {np.mean(apply_blend(single.mean(0),T.mean(0),0.87,60.0)-y):+.3f}")
    print("VERDICT:", "bagging can replace the tree" if gb >= 0.9*gt else
          "bagging does NOT recover the tree's gain -- keep the tree")
