"""Hybrid model: physics prediction plus a regularized correction on its residuals.

The correction absorbs whatever the simplified ODE omits -- axial dispersion,
temperature-dependent properties, geometry effects -- without letting a free-form
learner invent behaviour in regions where the physics already knows the answer.

Deliberately weak: a shallow, heavily-regularized learner on 150 rows. If the
physics fit is already at low RMSE there is very little signal left in the
residuals, and an over-eager correction will fit noise in the deterministic
simulator output and *hurt*. `blend_shrinkage` scales the correction down; CV
decides whether it earns its place at all.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor

from .data import FEATURE_COLUMNS, TARGET, add_physics_features, ode_inputs
from .physics import DEFAULT_STEPS, LOWER, UPPER, integrate, residuals


def make_corrector(seed: int = 0) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=600,
        max_depth=6,          # shallow on purpose -- this is a correction, not a model
        min_samples_leaf=6,
        max_features=0.5,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )


def make_hybrid_predict_fn(
    x_start,
    shrinkage: float = 1.0,
    n_steps: int = DEFAULT_STEPS,
    refit_physics: bool = True,
    corrector_factory=None,
):
    """Build a (train_df, predict_df, seed) -> predictions callable.

    Both the ODE parameters and the residual corrector are fit inside the fold,
    so cross-validating this function scores the whole pipeline rather than just
    its last stage.
    """
    x_start = np.asarray(x_start, dtype=float)

    def predict(train_df, predict_df, seed: int) -> np.ndarray:
        y = train_df[TARGET].to_numpy(dtype=float)

        x = x_start
        if refit_physics:
            res = least_squares(
                residuals,
                x0=x_start,
                bounds=(LOWER, UPPER),
                args=(ode_inputs(train_df), y, n_steps),
                x_scale="jac",
                xtol=1e-12, ftol=1e-12, gtol=1e-12,
                max_nfev=3000,
            )
            x = res.x

        phys_tr = integrate(x, ode_inputs(train_df), n_steps=n_steps)
        phys_te = integrate(x, ode_inputs(predict_df), n_steps=n_steps)

        if shrinkage == 0.0:
            return np.clip(phys_te, 0.0, 100.0)

        corrector = (corrector_factory or make_corrector)(seed)
        Xtr = add_physics_features(train_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
        corrector.fit(Xtr, y - phys_tr)
        Xte = add_physics_features(predict_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
        return np.clip(phys_te + shrinkage * corrector.predict(Xte), 0.0, 100.0)

    return predict
