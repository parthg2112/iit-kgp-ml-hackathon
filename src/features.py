"""Reaction-engineering features for the tree component of the blend.

Why this module exists. A tree makes axis-aligned splits on single features, so it
*cannot construct* `exp(-k1*tau) - exp(-k2*tau)` no matter how many rows it sees. That
expression is the closed-form solution of the isothermal series reaction, and it is most of
the physics of this problem. If we want the tree to be a useful ensemble member we have to
hand it that quantity rather than hope it rediscovers it from L, Q and T.

For an isothermal first-order series reaction A -> B -> C:

    Y_iso(tau)  = 100 * k1/(k2-k1) * (exp(-k1*tau) - exp(-k2*tau))
    tau_opt     = ln(k2/k1) / (k2-k1)            <- the interior optimum
    Y_max       = 100 * (k1/k2)^(k2/(k2-k1))     <- the ceiling at that temperature

`tau_opt` matters on its own: it is why corr(log tau, yield) is only +0.061. Residence time
has an interior optimum, so both tails are low-yield and the linear correlation cancels.
`log(tau/tau_opt)` converts that into a monotone signal a tree can actually split on.

LEAKAGE. These need rate constants, and using the *fitted* ones would leak information from
the rows a fold is scored on. We do not need them: measured correlation with the target is
+0.747 to +0.774 across a wide grid of trial kinetics, against +0.761 at the fitted values.
The information is in the functional form, not in the exact parameters. So the grid below is
fixed a priori -- same discipline as `TRIAL_E` in data.py -- and no fold-refitting is needed.

These features are ADDITIVE. `add_physics_features` in data.py is deliberately untouched so
every existing artifact stays reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, R_GAS, T_REF, add_physics_features

# Trial kinetics: (ln_k1_ref, E1_kJ, ln_k2_ref, E2_kJ), rate constants referenced to T_REF.
# Chosen a priori to span plausible reaction-engineering values -- E1 in 40-80 kJ/mol for a
# moderate liquid-phase reaction, E2 higher for a more temperature-sensitive degradation --
# NOT read off our fit. None of these tuples equals the fitted parameters.
TRIAL_KINETICS: tuple[tuple[float, float, float, float], ...] = (
    (2.5, 40.0, 0.0, 150.0),
    (2.5, 40.0, 0.0, 250.0),
    (3.0, 60.0, 0.5, 200.0),
    (2.0, 80.0, -0.5, 280.0),
)

_TINY = 1e-12


def _rate(ln_k_ref: float, E_kJ: float, T: np.ndarray) -> np.ndarray:
    return np.exp(ln_k_ref - (E_kJ * 1000.0 / R_GAS) * (1.0 / T - 1.0 / T_REF))


def series_yield(k1: np.ndarray, k2: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Closed-form isothermal yield (%), with the confluent k1 -> k2 limit handled."""
    d = k2 - k1
    near = np.abs(d) < _TINY
    d_safe = np.where(near, 1.0, d)
    with np.errstate(over="ignore", invalid="ignore"):
        general = (k1 / d_safe) * (np.exp(-k1 * tau) - np.exp(-k2 * tau))
        confluent = k1 * tau * np.exp(-k1 * tau)          # limit as k2 -> k1
        y = 100.0 * np.where(near, confluent, general)
    return np.clip(np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)


def optimum(k1: np.ndarray, k2: np.ndarray):
    """(tau_opt, Y_max) for the isothermal series reaction."""
    d = k2 - k1
    near = np.abs(d) < _TINY
    d_safe = np.where(near, 1.0, d)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = np.clip(k2 / np.maximum(k1, _TINY), _TINY, 1e12)
        tau_opt = np.where(near, 1.0 / np.maximum(k1, _TINY), np.log(ratio) / d_safe)
        y_max = np.where(near, 100.0 / np.e,
                         100.0 * np.power(np.clip(k1 / np.maximum(k2, _TINY), _TINY, 1e12),
                                          k2 / d_safe))
    tau_opt = np.clip(np.nan_to_num(tau_opt, nan=1.0, posinf=1e6, neginf=_TINY), _TINY, 1e6)
    y_max = np.clip(np.nan_to_num(y_max, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
    return tau_opt, y_max


def add_reaction_features(df: pd.DataFrame,
                          kinetics=TRIAL_KINETICS) -> pd.DataFrame:
    """Append the reaction-engineering block on top of `add_physics_features`."""
    out = add_physics_features(df)
    T = out["T_avg"].to_numpy(dtype=float)
    tau = out["tau"].to_numpy(dtype=float)
    CA0 = out["concentration_mol_L"].to_numpy(dtype=float)

    for i, (ln_k1, E1, ln_k2, E2) in enumerate(kinetics):
        k1, k2 = _rate(ln_k1, E1, T), _rate(ln_k2, E2, T)
        tau_opt, y_max = optimum(k1, k2)
        y_iso = series_yield(k1, k2, tau)

        # The solution of the simplified problem, handed over whole.
        out[f"Yiso_{i}"] = y_iso
        # Where this row sits relative to the interior optimum. This is the feature that
        # rescues residence time from its near-zero linear correlation.
        out[f"logtau_rel_{i}"] = np.log(np.clip(tau / tau_opt, 1e-6, 1e6))
        # Fraction of the achievable ceiling reached at this temperature.
        out[f"Yfrac_{i}"] = y_iso / np.maximum(y_max, _TINY)
        # True Damkohler numbers -- ln_Da_E in data.py drops the pre-exponential, so it is
        # only a relative group; these are the real ones.
        out[f"lnDa1_{i}"] = np.log(np.clip(k1 * tau, 1e-12, 1e12))
        out[f"lnDa2_{i}"] = np.log(np.clip(k2 * tau, 1e-12, 1e12))
        # Selectivity ratio: > 0 means the waste reaction is outrunning the desired one.
        out[f"ln_sel_{i}"] = np.log(np.clip(k2 / np.maximum(k1, _TINY), 1e-12, 1e12))

    # Thermal-path groups that need no fitted parameter. Concentration enters yield only
    # through reaction heat, so it is only ever informative in combination with temperature.
    out["CA0_x_dT"] = CA0 * out["delta_T"].to_numpy(dtype=float)
    out["T_avg_rel"] = T - T_REF
    return out


def reaction_feature_columns(kinetics=TRIAL_KINETICS) -> list[str]:
    """Full column list: the existing physics features plus this block."""
    cols = list(FEATURE_COLUMNS)
    for i in range(len(kinetics)):
        cols += [f"Yiso_{i}", f"logtau_rel_{i}", f"Yfrac_{i}",
                 f"lnDa1_{i}", f"lnDa2_{i}", f"ln_sel_{i}"]
    return cols + ["CA0_x_dT", "T_avg_rel"]
