"""Non-isothermal plug-flow reactor model: A --k1--> B --k2--> C.

Governing system, integrated along the reactor axis z from 0 to tau:

    dCA/dz = -k1*CA
    dCB/dz =  k1*CA - k2*CB
    dT/dz  =  a1*k1*CA + a2*k2*CB + U*(T_jacket - T)

with CA(0) = CA0, CB(0) = 0, T(0) = T_inlet, and yield = 100 * CB(tau) / CA0.

Two implementation choices matter enough to spell out:

1.  Arrhenius is reparameterized about a reference temperature:

        k = exp(ln_k_ref) * exp(-E/R * (1/T - 1/T_ref))

    Fitting ln(A) and E directly makes the two parameters correlated >0.999 --
    the classic long narrow valley that makes this fit "stiff and slow to
    converge". Centering on T_ref ~ mean(T) decorrelates them.

2.  All rows are integrated *simultaneously* as numpy arrays of shape (n,) with
    a fixed-step RK4 over a normalized axis s in [0, 1] (z = s*tau). This makes
    one residual evaluation a few hundred vector ops instead of 150 separate
    solve_ivp calls, which is what makes multistart affordable.

Concentration enters only through the thermal path (a1*CA0, a2*CA0). If the fit
returns a1, a2 ~ 0 then the reactions are near thermally neutral, which is the
quantitative explanation for the observed corr(concentration, yield) = +0.009.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import R_GAS, T_REF

# Parameter order used throughout. E1/E2 are carried in kJ/mol so every entry of
# the vector is O(1)-O(200); this keeps the least-squares Jacobian well scaled.
PARAM_NAMES = ("ln_k1_ref", "E1_kJ", "ln_k2_ref", "E2_kJ", "a1", "a2", "U", "n_flow")

# Reference flow for the U(Q) correlation, ~ the data mean.
Q_REF = 40.0  # L/min

# Physically motivated box constraints.
#
# a1/a2 are (-dH)/(rho*Cp) in K*L/mol, so a*CA0 is the adiabatic temperature
# rise of that reaction. CA0 reaches 3.97 mol/L, so |a| <= 30 caps each
# reaction's adiabatic excursion at ~120 K. Wider bounds let the optimizer walk
# into thermal-runaway corners where no affordable step count converges, and it
# then fits the *integration error* rather than the data.
#
# n_flow is the exponent in U_eff = U*(Q/Q_REF)^n_flow. In the z = L/Q
# coordinate a *constant* physical U gives a flow-independent jacket term, so
# n_flow = 0 is the plain plug-flow model. But physical U tracks the Reynolds
# number (~Q^0.8 for turbulent internal flow), which is exactly the effect
# guide.md describes as flow rate affecting "how much heat the jacket can
# transfer per unit of fluid". Turbulent correlations put it near 0.8.
LOWER = np.array([-12.0, 40.0, -12.0, 60.0, -30.0, -30.0, 0.0, -1.5])
UPPER = np.array([12.0, 140.0, 12.0, 280.0, 30.0, 30.0, 60.0, 1.5])

# Model variants. Each maps a parameter name to the value it is pinned at;
# everything not listed is free. Comparing these is how we decide which
# mechanism the data actually supports, rather than assuming one.
MODELS: dict[str, dict[str, float]] = {
    # Plain non-isothermal plug flow, constant U -- the baseline form.
    "series": {"n_flow": 0.0},
    # Same, but with both reactions forced thermally neutral. If this matches
    # `series`, the heat terms were never doing real work and concentration
    # genuinely drops out of the problem.
    "neutral": {"a1": 0.0, "a2": 0.0, "n_flow": 0.0},
    # Flow-dependent jacket heat transfer.
    "flowU": {},
    "flowU_neutral": {"a1": 0.0, "a2": 0.0},
}


@dataclass(frozen=True)
class ReactorParams:
    ln_k1_ref: float
    E1_kJ: float
    ln_k2_ref: float
    E2_kJ: float
    a1: float
    a2: float
    U: float
    n_flow: float = 0.0

    @classmethod
    def from_vector(cls, x) -> "ReactorParams":
        # Vectors saved before n_flow existed are 7 long; treat them as n_flow=0.
        vals = [float(v) for v in x]
        if len(vals) == len(PARAM_NAMES) - 1:
            vals.append(0.0)
        return cls(*vals)

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in PARAM_NAMES}


def _rate(ln_k_ref: float, E_kJ: float, T: np.ndarray) -> np.ndarray:
    """Arrhenius rate constant, reparameterized about T_REF."""
    return np.exp(ln_k_ref - (E_kJ * 1000.0 / R_GAS) * (1.0 / T - 1.0 / T_REF))


# Substeps for reported/final predictions. 256 is used during the global search
# (~11 ms per full 150-row evaluation); 512 costs ~22 ms and is what final
# numbers are quoted at. Convergence is re-asserted at the fitted parameters.
DEFAULT_STEPS = 512
SEARCH_STEPS = 256

# Substeps for the one scored artifact. At the fitted parameters individual
# predictions still move by ~0.22 yield-points between 512 and 2048 substeps,
# and the finer grid is the better one (train RMSE 3.6559 at 2048 vs 3.6617 at
# 512), so this is free accuracy rather than a tradeoff. ~90 ms, paid once.
# Both make_submission.py and the notebook use this -- keep them in sync or the
# next notebook execution silently rewrites the CSV at the coarser grid.
SUBMIT_STEPS = 2048


def _series_step(xA, xB, k1, k2, dz):
    """Exact solution of the linear series reaction over dz at frozen k1, k2.

        xA' = xA*exp(-k1*dz)
        xB' = xB*exp(-k2*dz) + xA * k1/(k2-k1) * (exp(-k1*dz) - exp(-k2*dz))

    This is what makes the scheme unconditionally stable: however large k*dz
    gets, the exponentials decay to zero instead of oscillating. A fixed-step
    RK4 blows up here once k*dz exceeds ~2.8.
    """
    e1 = np.exp(-k1 * dz)
    e2 = np.exp(-k2 * dz)
    d = k2 - k1
    # Removable singularity at k1 == k2: fall back to the confluent limit
    # xB' = xB*e1 + xA*k1*dz*e1.
    near = np.abs(d * dz) < 1e-8
    d_safe = np.where(near, 1.0, d)
    gain = np.where(near, k1 * dz * e1, k1 / d_safe * (e1 - e2))
    xA_new = xA * e1
    xB_new = xB * e2 + xA * gain
    return xA_new, xB_new


def _temperature_step(T, T_jacket, Q, U, dz):
    """Exact solution of dT/dz = S + U*(T_jacket - T) over dz, S = Q/dz constant.

    Q is the total heat released over the step (in temperature units), so the
    jacket exchange stays exact however large U*dz becomes. U may be a scalar or
    a per-row array (the flow-dependent correlation makes it the latter).
    """
    Udz = np.asarray(U) * dz
    tiny = Udz < 1e-12
    # Guard the division so the U -> 0 branch stays finite; `where` picks it.
    S_over_U = np.where(tiny, 0.0, Q / np.where(tiny, 1.0, Udz))
    exact = T_jacket + S_over_U + (T - T_jacket - S_over_U) * np.exp(-Udz)
    limit = T + Q + (T_jacket - T) * Udz
    return np.where(tiny, limit, exact)


def integrate(x, inputs: dict[str, np.ndarray], n_steps: int = DEFAULT_STEPS) -> np.ndarray:
    """Integrate every row at once and return predicted yield (%) per row.

    Second-order operator splitting. Within a substep the rate constants are
    frozen, which lets the mass balance advance *analytically* (`_series_step`)
    and the energy balance advance with the exact linear solution
    (`_temperature_step`) -- both unconditionally stable, so no step size can
    make concentrations go negative or temperature oscillate. Freezing k is the
    only source of error, and a midpoint predictor-corrector makes that error
    second order in the step size rather than first.

    Parameters
    ----------
    x : parameter vector in PARAM_NAMES order
    inputs : dict from data.ode_inputs -- CA0, T_in, T_jacket, tau arrays
    n_steps : substeps along the normalized axis
    """
    p = ReactorParams.from_vector(x)
    CA0 = inputs["CA0"]
    T_jacket = inputs["T_jacket"]
    tau = inputs["tau"]

    dz = tau / n_steps  # physical step along the reactor axis, per row
    # U_eff is per-row once the flow correlation is active.
    U = p.U if p.n_flow == 0.0 else p.U * (inputs["Q"] / Q_REF) ** p.n_flow
    half = 0.5 * dz

    def heat(xA, xB, xA_new, xB_new):
        consumed_A = xA - xA_new                    # all of it became B
        consumed_B = consumed_A - (xB_new - xB)     # B produced minus B accumulated
        return p.a1 * CA0 * consumed_A + p.a2 * CA0 * consumed_B

    xA = np.ones_like(CA0)
    xB = np.zeros_like(CA0)
    T = inputs["T_in"].copy()

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(n_steps):
            # Predictor: half a step at the current temperature, to locate the
            # midpoint temperature where the rates should really be evaluated.
            k1 = _rate(p.ln_k1_ref, p.E1_kJ, T)
            k2 = _rate(p.ln_k2_ref, p.E2_kJ, T)
            xA_h, xB_h = _series_step(xA, xB, k1, k2, half)
            T_mid = _temperature_step(T, T_jacket, heat(xA, xB, xA_h, xB_h), U, half)

            # Corrector: full step using the midpoint rates.
            k1m = _rate(p.ln_k1_ref, p.E1_kJ, T_mid)
            k2m = _rate(p.ln_k2_ref, p.E2_kJ, T_mid)
            xA_new, xB_new = _series_step(xA, xB, k1m, k2m, dz)
            T = _temperature_step(T, T_jacket, heat(xA, xB, xA_new, xB_new), U, dz)

            xA, xB = xA_new, xB_new

    y = 100.0 * xB
    return np.where(np.isfinite(y), np.clip(y, 0.0, 100.0), 0.0)


def predict(x, df, n_steps: int = DEFAULT_STEPS) -> np.ndarray:
    """Convenience wrapper: DataFrame in, yield predictions out."""
    from .data import ode_inputs

    return integrate(x, ode_inputs(df), n_steps=n_steps)


# Parameter sets whose integration has not converged at the working step count
# get this residual instead of their (meaningless) predictions. Without it the
# optimizer happily minimizes integration error in stiff corners -- observed as
# a fit whose predictions moved by 92 yield-points when substeps were increased.
POISON = 1e3
CONVERGENCE_TOL = 0.05


def residuals(
    x,
    inputs: dict[str, np.ndarray],
    y: np.ndarray,
    n_steps: int = DEFAULT_STEPS,
    convergence_tol: float | None = None,
) -> np.ndarray:
    """Prediction error per row, with an optional convergence guard.

    The guard defaults **off**. It belongs in the global search, where it stops
    differential evolution wandering into stiff corners, and it must be passed
    explicitly there (`convergence_tol=CONVERGENCE_TOL`). It must NOT be on
    during a `least_squares` polish: the poison value is a cliff in an otherwise
    smooth objective, and a finite-difference Jacobian straddling that cliff is
    garbage. Measured directly -- the guard fires 159/256 times around the
    optimum, and turning it off during polish moved the fit from 4.38 to 3.66.
    """
    pred = integrate(x, inputs, n_steps=n_steps)

    if convergence_tol is not None:
        # Half-resolution comparison costs ~50% more per evaluation and makes
        # every accepted parameter set one whose numerics are actually resolved.
        coarse = integrate(x, inputs, n_steps=max(8, n_steps // 2))
        if np.abs(pred - coarse).max() > convergence_tol:
            return np.full_like(y, POISON, dtype=float)

    r = pred - y
    # A diverged parameter set must not look better than a bad-but-finite one.
    return np.where(np.isfinite(r), r, POISON)


def objective(
    x,
    inputs: dict[str, np.ndarray],
    y: np.ndarray,
    n_steps: int = DEFAULT_STEPS,
    convergence_tol: float | None = None,
) -> float:
    r = residuals(x, inputs, y, n_steps=n_steps, convergence_tol=convergence_tol)
    return float(np.sqrt(np.mean(r**2)))


class MaskedModel:
    """Restricts the 8-parameter vector to the free parameters of one variant.

    `MODELS[name]` pins some parameters to fixed values; this exposes only the
    remaining ones to the optimizer, so a 5-parameter variant is genuinely
    searched in 5 dimensions instead of being nudged toward a pinned value.
    """

    def __init__(self, name: str):
        if name not in MODELS:
            raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
        self.name = name
        self.pinned = MODELS[name]
        self.free_idx = [i for i, n in enumerate(PARAM_NAMES) if n not in self.pinned]
        self.free_names = [PARAM_NAMES[i] for i in self.free_idx]
        self.lower = LOWER[self.free_idx]
        self.upper = UPPER[self.free_idx]

    def expand(self, x_free) -> np.ndarray:
        """Free-parameter vector -> full 8-parameter vector."""
        full = np.zeros(len(PARAM_NAMES))
        for name, value in self.pinned.items():
            full[PARAM_NAMES.index(name)] = value
        full[self.free_idx] = np.asarray(x_free, dtype=float)
        return full

    def restrict(self, x_full) -> np.ndarray:
        return np.asarray(x_full, dtype=float)[self.free_idx]

    def residuals(self, x_free, inputs, y, n_steps=DEFAULT_STEPS, convergence_tol=None):
        return residuals(self.expand(x_free), inputs, y, n_steps, convergence_tol)

    def objective(self, x_free, inputs, y, n_steps=DEFAULT_STEPS, convergence_tol=None):
        return objective(self.expand(x_free), inputs, y, n_steps, convergence_tol)


# Module-level wrappers so multiprocessing (differential_evolution workers=-1)
# can pickle them; bound methods of a locally-built object cannot be pickled.
def masked_objective(x_free, model_name, inputs, y, n_steps, convergence_tol):
    return MaskedModel(model_name).objective(x_free, inputs, y, n_steps, convergence_tol)


def masked_residuals(x_free, model_name, inputs, y, n_steps, convergence_tol):
    return MaskedModel(model_name).residuals(x_free, inputs, y, n_steps, convergence_tol)


# --- Arbitrary parameter pinning, picklable ------------------------------------
# MODELS covers the named variants, but profiling needs a parameter pinned at an
# arbitrary value. These are module-level (not closures) so differential_evolution
# can ship them to worker processes -- a lambda here silently forces the whole fit
# onto a single core.

def pin_expand(x_free, pin_names, pin_values) -> np.ndarray:
    """Free-parameter vector plus pinned values -> full parameter vector."""
    full = np.zeros(len(PARAM_NAMES))
    for name, value in zip(pin_names, pin_values):
        full[PARAM_NAMES.index(name)] = value
    free = [i for i, n in enumerate(PARAM_NAMES) if n not in pin_names]
    full[free] = np.asarray(x_free, dtype=float)
    return full


def pin_free_indices(pin_names):
    return [i for i, n in enumerate(PARAM_NAMES) if n not in pin_names]


def pinned_objective(x_free, pin_names, pin_values, inputs, y, n_steps, convergence_tol):
    return objective(pin_expand(x_free, pin_names, pin_values), inputs, y, n_steps, convergence_tol)


def pinned_residuals(x_free, pin_names, pin_values, inputs, y, n_steps, convergence_tol):
    return residuals(pin_expand(x_free, pin_names, pin_values), inputs, y, n_steps, convergence_tol)


def reference_solve(x, inputs: dict[str, np.ndarray], indices) -> np.ndarray:
    """Independent check of `integrate` using scipy's stiff BDF solver.

    Used only in tests -- one row at a time, no vectorization, no clamping.
    """
    from scipy.integrate import solve_ivp

    p = ReactorParams.from_vector(x)
    out = []
    for i in indices:
        CA0 = inputs["CA0"][i]
        Tj = inputs["T_jacket"][i]
        tau_i = inputs["tau"][i]
        U_i = p.U * (inputs["Q"][i] / Q_REF) ** p.n_flow

        def rhs(_z, s, U_i=U_i, CA0=CA0, Tj=Tj):
            xA, xB, T = s
            k1 = float(_rate(p.ln_k1_ref, p.E1_kJ, np.array([T]))[0])
            k2 = float(_rate(p.ln_k2_ref, p.E2_kJ, np.array([T]))[0])
            return [
                -k1 * xA,
                k1 * xA - k2 * xB,
                p.a1 * CA0 * k1 * xA + p.a2 * CA0 * k2 * xB + U_i * (Tj - T),
            ]

        sol = solve_ivp(
            rhs, (0.0, tau_i), [1.0, 0.0, inputs["T_in"][i]],
            method="BDF", rtol=1e-10, atol=1e-12,
        )
        out.append(100.0 * sol.y[1, -1])
    return np.array(out)
