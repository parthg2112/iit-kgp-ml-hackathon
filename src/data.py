"""Data loading, physics feature engineering, and submission writing.

The five raw inputs are operating knobs on a non-isothermal plug-flow reactor.
The derived features here are dimensionless / reaction-engineering groups, not
generic polynomial combinations -- see README of guide.md section 5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "train_dataset.csv"
TEST_CSV = DATA_DIR / "test_dataset.csv"

RAW_FEATURES = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
]
TARGET = "overall_yield"

R_GAS = 8.314  # J/(mol K)

# Trial activation energies for the log-Damkohler features. Spanning a plausible
# range lets a tree pick whichever is closest to the true E1/E2 without us
# having to know them in advance.
TRIAL_E = (60_000.0, 100_000.0, 160_000.0)

# Reference temperature for Arrhenius reparameterization, ~ the data mean.
# Fitting ln(A) and E directly makes them correlated >0.999, which turns the
# least-squares surface into a long narrow valley. Centering on T_ref removes it.
T_REF = 430.0  # K


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append reaction-engineering derived features.

    tau = length / flow_rate is proportional to residence time. Its units are
    m*min/L, not time -- the true tau is A_c*L/Q. Tube diameter is constant
    across rows, so A_c folds into the fitted pre-exponentials and into U.
    """
    out = df.copy()
    tau = df["length_m"] / df["flow_rate_L_min"]
    out["tau"] = tau
    out["log_tau"] = np.log(tau)
    out["T_avg"] = (df["inlet_temperature_K"] + df["jacket_temperature_K"]) / 2.0
    out["delta_T"] = df["jacket_temperature_K"] - df["inlet_temperature_K"]
    out["inv_T_avg"] = 1.0 / out["T_avg"]

    for e in TRIAL_E:
        # ln(Da) = ln(tau) - E/(R*T). The dimensionless group governing conversion.
        out[f"ln_Da_{int(e / 1000)}k"] = out["log_tau"] - e / (R_GAS * out["T_avg"])

    return out


FEATURE_COLUMNS = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
    "tau",
    "log_tau",
    "T_avg",
    "delta_T",
    "inv_T_avg",
    *[f"ln_Da_{int(e / 1000)}k" for e in TRIAL_E],
]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return add_physics_features(df)[FEATURE_COLUMNS].to_numpy(dtype=float)


def ode_inputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The four arrays the reactor integrator needs, one element per row."""
    return {
        "CA0": df["concentration_mol_L"].to_numpy(dtype=float),
        "T_in": df["inlet_temperature_K"].to_numpy(dtype=float),
        "T_jacket": df["jacket_temperature_K"].to_numpy(dtype=float),
        "tau": (df["length_m"] / df["flow_rate_L_min"]).to_numpy(dtype=float),
        # Needed only by the flow-dependent heat-transfer correlation.
        "Q": df["flow_rate_L_min"].to_numpy(dtype=float),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def write_submission(predictions: np.ndarray, team_name: str, out_dir: Path | None = None) -> Path:
    """Write the one-and-only submission file, validating the contract first.

    Contract (problem statement section 5): exactly 50 rows in test_dataset.csv
    order, exactly one column headed `overall_yield`, floats to >=3 decimals,
    no index column, file named [TeamName].csv.
    """
    preds = np.asarray(predictions, dtype=float).ravel()
    test = load_test()

    assert preds.shape == (len(test),), f"expected {len(test)} predictions, got {preds.shape}"
    assert len(test) == 50, f"test set should have 50 rows, has {len(test)}"
    assert np.all(np.isfinite(preds)), "predictions contain NaN or inf"
    assert preds.min() >= 0.0 and preds.max() <= 100.0, (
        f"predictions outside [0, 100]: [{preds.min()}, {preds.max()}]"
    )

    out_dir = out_dir or ROOT
    path = out_dir / f"{team_name}.csv"
    frame = pd.DataFrame({TARGET: np.round(preds, 6)})
    frame.to_csv(path, index=False, float_format="%.6f")

    # Read back and re-validate what actually landed on disk.
    check = pd.read_csv(path)
    assert list(check.columns) == [TARGET], f"bad columns: {list(check.columns)}"
    assert len(check) == 50, f"bad row count: {len(check)}"
    return path
