"""ExtraTrees safety net on physics features.

ExtraTrees rather than boosting: at n=150 the extra randomization in split
selection is the regularizer this problem needs, and gradient boosting memorizes
the training set before it learns the shape of the yield surface.

This is insurance, not the main event -- it cannot represent the sharp yield
cliff smoothly, because a step function approximated by piecewise constants
always leaves residual error at the transition band.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .data import FEATURE_COLUMNS, add_physics_features


def make_model(seed: int = 0) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=800,
        max_features=0.6,
        min_samples_leaf=1,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )


def fit_predict(train_df, predict_df, seed: int = 0) -> np.ndarray:
    """Train on `train_df` (must contain the target) and predict `predict_df`."""
    from .data import TARGET

    model = make_model(seed)
    X = add_physics_features(train_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
    model.fit(X, train_df[TARGET].to_numpy(dtype=float))
    Xp = add_physics_features(predict_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
    return np.clip(model.predict(Xp), 0.0, 100.0)
