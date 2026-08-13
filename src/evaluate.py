"""Cross-validation harness and blend-weight search.

At n=150 a single train/test split is noise: swapping the seed moves RMSE by
several points. Everything here is repeated K-fold across multiple seeds, and
the numbers quoted are mean +/- std over seeds.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

from .data import TARGET, ode_inputs, rmse
from .physics import DEFAULT_STEPS, LOWER, UPPER, integrate, residuals

# Signature: (train_df, predict_df, seed) -> predictions for predict_df
PredictFn = Callable[[pd.DataFrame, pd.DataFrame, int], np.ndarray]


def oof_predictions(predict_fn: PredictFn, df: pd.DataFrame, seed: int, n_splits: int = 10) -> np.ndarray:
    """Out-of-fold predictions for one seed. Every fold refits from scratch."""
    oof = np.full(len(df), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kf.split(df):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        oof[te_idx] = predict_fn(tr, te, seed)
    assert not np.isnan(oof).any(), "some rows never landed in a validation fold"
    return oof


def repeated_cv(
    predict_fn: PredictFn,
    df: pd.DataFrame,
    seeds=(0, 1, 2, 3, 4),
    n_splits: int = 10,
) -> tuple[np.ndarray, dict]:
    """Returns (oof matrix of shape (n_seeds, n_rows), summary stats)."""
    y = df[TARGET].to_numpy(dtype=float)
    mat = np.vstack([oof_predictions(predict_fn, df, seed, n_splits) for seed in seeds])
    per_seed = [rmse(y, row) for row in mat]
    return mat, {
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed)),
        "per_seed": per_seed,
        "seeds": list(seeds),
    }


def make_physics_predict_fn(x_start, n_steps: int = DEFAULT_STEPS) -> PredictFn:
    """Physics predictor that *refits the 7 parameters inside each fold*.

    Caveat worth stating out loud: `x_start` comes from the full-data global
    search, so the starting point carries a little information from the held-out
    rows. Running differential evolution inside every fold would remove that but
    costs ~50x more. The parameters themselves are refit on fold-train only, and
    with 7 parameters against 135 rows the residual optimism is small -- but it
    means fold RMSE is a mild lower bound, not an unbiased estimate.
    """
    x_start = np.asarray(x_start, dtype=float)

    def predict(train_df: pd.DataFrame, predict_df: pd.DataFrame, seed: int) -> np.ndarray:
        inputs = ode_inputs(train_df)
        y = train_df[TARGET].to_numpy(dtype=float)
        res = least_squares(
            residuals,
            x0=x_start,
            bounds=(LOWER, UPPER),
            args=(inputs, y, n_steps),
            x_scale="jac",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=3000,
        )
        return integrate(res.x, ode_inputs(predict_df), n_steps=n_steps)

    return predict


# Above this physics prediction the tree is dropped entirely. Trees cannot
# extrapolate past their outermost split, so on high-yield rows they can only pull
# predictions toward the training mean -- measured as a one-directional loss
# (97.7 -> 96.6, 93.0 -> 91.8, 95.1 -> 94.7). Below the cutoff, near the yield
# cliff, the ODE carries its largest bias and local interpolation genuinely helps.
# Validated leave-one-seed-out: 6.022 vs 6.105 for a flat blend, better on all
# three held-out seeds.
BLEND_CUTOFF = 60.0


def apply_blend(physics, tree, weight: float, cutoff: float = BLEND_CUTOFF) -> np.ndarray:
    """Regime-aware blend: mix below `cutoff`, pure physics above it."""
    physics = np.asarray(physics, dtype=float)
    mixed = np.clip(weight * physics + (1.0 - weight) * np.asarray(tree, dtype=float), 0.0, 100.0)
    return np.where(physics <= cutoff, mixed, np.clip(physics, 0.0, 100.0))


def best_blend_weight(y: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray, n_grid: int = 2001):
    """Pick w minimizing RMSE of w*a + (1-w)*b on out-of-fold predictions.

    A fixed 70/30 split is the wrong instinct when the two models are far apart
    in accuracy: if `a` scores 2 and `b` scores 14, 70/30 lands near 4.4 -- more
    than double the error, paid as "insurance". Search the weight instead.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    scores = np.array([rmse(y, np.clip(w * oof_a + (1 - w) * oof_b, 0, 100)) for w in grid])
    i = int(np.argmin(scores))
    return float(grid[i]), float(scores[i]), grid, scores
