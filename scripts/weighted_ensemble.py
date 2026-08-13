"""Is a fit-weighted average over admissible parameter sets better than the best fit?

A reviewer suggested predicting the "posterior mean" over the admissible set rather than
the value at the single best fit, on the grounds that under squared error the mean of a
genuine posterior beats a point estimate.

That reasoning is right in general but its premise needs checking here. Under a Gaussian
likelihood with n=150, going from RMSE 3.66 to 3.99 is a log-likelihood drop of about 13,
i.e. a weight ratio near 2.5e-6 -- so a correctly weighted posterior collapses onto the best
fit and the "average" is just the point estimate again. The 10%-RMSE admissible band used
for profiling is a display convention, not a statement that those points are equally likely.

The counter-argument is that our model is *misspecified*: the data is deterministic, so a
train RMSE of 3.66 is model-form error, not observation noise. Under misspecification the
naive likelihood is overconfident and some spread is warranted -- but how much is an
empirical question, not something to assume.

So: sweep the weighting sharpness from flat to likelihood-exact, and validate each
out-of-fold. Adopt only on a positive result. A null result is a finding, not a failure.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_test, load_train, ode_inputs, rmse
from src.physics import DEFAULT_STEPS, LOWER, SUBMIT_STEPS, UPPER, integrate, residuals

ARTIFACTS = ROOT / "artifacts"
N_ROWS = 150
# 0 = flat average over admissible sets; 1 = exact Gaussian likelihood weighting.
TEMPERATURES = [0.0, 0.05, 0.15, 0.35, 1.0]


def collect():
    """Every fitted parameter vector on disk, with its train RMSE."""
    out = []
    for fname, extract in [
        ("physics_params.json", lambda d: [(d["vector"], d["train_rmse"])]),
        ("model_comparison.json", lambda d: [(v["vector"], v["train_rmse"]) for v in d.values()]),
        ("cold_folds.json", lambda d: [(v, None) for v in d["fold_vectors"]]),
        ("wide_bounds.json", lambda d: [(d["vector"], d["train_rmse"])]),
        ("profile_E2_kJ.json", lambda d: [(p["vector"], p["train_rmse"]) for p in d["points"].values()]),
        ("profile_a1.json", lambda d: [(p["vector"], p["train_rmse"]) for p in d["points"].values()]),
        ("profile_n_flow.json", lambda d: [(p["vector"], p["train_rmse"]) for p in d["points"].values()]),
    ]:
        p = ARTIFACTS / fname
        if p.exists():
            out.extend(extract(json.loads(p.read_text())))
    return out


def weights(rmses, temperature):
    """Gaussian-likelihood weights, flattened by `temperature` (1 = exact, 0 = flat)."""
    r = np.asarray(rmses, dtype=float)
    if temperature <= 0:
        return np.ones_like(r) / len(r)
    # log L ~ -n/2 * log(rmse^2); temperature scales the sharpness.
    ll = -temperature * (N_ROWS / 2.0) * np.log(r ** 2)
    ll -= ll.max()
    w = np.exp(ll)
    return w / w.sum()


def main() -> int:
    tr, te = load_train(), load_test()
    y = tr[TARGET].to_numpy(dtype=float)
    inp_tr, inp_te = ode_inputs(tr), ode_inputs(te)

    vecs = []
    for v, _ in collect():
        v = np.clip(np.array(v, dtype=float), LOWER, UPPER)
        vecs.append((rmse(y, integrate(v, inp_tr, n_steps=DEFAULT_STEPS)), v))
    vecs.sort(key=lambda t: t[0])
    best = vecs[0][0]
    keep = [(s, v) for s, v in vecs if s <= best * 1.25]
    print(f"{len(vecs)} fitted parameter sets on disk; {len(keep)} within 1.25x of best "
          f"({best:.4f})")
    print(f"  their train RMSEs: {np.round([s for s, _ in keep], 3)}\n")

    R = np.array([s for s, _ in keep])
    for t in TEMPERATURES:
        w = weights(R, t)
        print(f"  temperature {t:4.2f}: top weight {w.max():.4f}, effective n = {1/np.sum(w**2):.2f}")

    # --- Degeneracy test, which decides the question outright ---------------
    #
    # An ensemble can only help if its members stay distinct when refit. The
    # apparent diversity here comes from *pinning* during profiling; a free refit
    # removes the pin. So before running a 3-seed x 10-fold x 8-start validation
    # (~70 minutes, and it was serial), check whether the members survive at all.
    print("\ndegeneracy test: refit 8 distinct starts on a single fold")
    starts = [v for _, v in keep[: min(8, len(keep))]]
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    tr_idx, te_idx = next(iter(kf.split(tr)))
    sub = tr.iloc[tr_idx]
    inp_s = ode_inputs(sub)
    ysub = sub[TARGET].to_numpy(dtype=float)

    fits = []
    for i, x0 in enumerate(starts):
        r = least_squares(residuals, x0=x0, bounds=(LOWER, UPPER),
                          args=(inp_s, ysub, DEFAULT_STEPS, None), x_scale="jac",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        fits.append(r.x)
        print(f"  start {i}: E2 {x0[3]:7.2f} -> refit E2 {r.x[3]:7.2f}, "
              f"a1 {r.x[4]:7.3f}, fold RMSE {rmse(ysub, integrate(r.x, inp_s)):.4f}")

    F = np.array(fits)
    spread = float(F.std(0).max())
    print(f"\nmax parameter std across the 8 refits: {spread:.2e}")

    degenerate = spread < 1e-3
    if degenerate:
        print("VERDICT: keep the single fit.")
        print("  Every admissible start converges to the SAME fold optimum, so any")
        print("  weighting of them -- flat or likelihood-sharp -- returns exactly the")
        print("  single-fit prediction. There is no posterior spread to average over.")
        print("  This matches the likelihood argument: at n=150, RMSE 3.66 vs 3.99 is a")
        print("  weight ratio near 2.5e-6, so the posterior was always going to be tight.")
        print("  Reported as a robustness result, not a failure: the objective has one")
        print("  well-defined optimum that is found reliably from any admissible start.")
    else:
        print("VERDICT: members stay distinct -- run the full out-of-fold validation.")

    (ARTIFACTS / "weighted_ensemble.json").write_text(json.dumps({
        "n_sets_considered": len(vecs),
        "n_within_1.25x": len(keep),
        "refit_param_spread": spread,
        "degenerate": bool(degenerate),
        "adopt": False if degenerate else None,
        "refit_optimum": [float(v) for v in F[0]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
