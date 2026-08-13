"""Verify the vectorized RK4 integrator against scipy's stiff BDF solver.

Gate: max absolute yield difference across sampled rows must be < 0.01 (%).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_train, ode_inputs
from src.physics import integrate, reference_solve

rng = np.random.default_rng(0)
df = load_train()
inputs = ode_inputs(df)

# Several parameter sets spanning slow, balanced, and violently fast regimes.
PARAM_SETS = {
    "slow":      [-1.0,  70.0, -2.5, 150.0,   0.0,   0.0,  0.5],
    "balanced":  [ 0.5,  80.0, -0.5, 160.0,  10.0, -10.0,  2.0],
    "fast":      [ 2.5,  90.0,  1.5, 180.0,   0.0,   0.0,  5.0],
    "exothermic":[ 0.5,  80.0, -0.5, 160.0,  60.0,  40.0,  1.0],
}

idx = rng.choice(len(df), size=10, replace=False)
worst_bdf = 0.0
worst_conv = 0.0

print("accuracy vs scipy BDF (10 sampled rows):")
for name, x in PARAM_SETS.items():
    fast = integrate(x, inputs)[idx]
    ref = reference_solve(x, inputs, idx)
    diff = np.abs(fast - ref)
    worst_bdf = max(worst_bdf, diff.max())
    print(f"  {name:11s} max|split-bdf| = {diff.max():.3e}   yields {fast.min():.2f}..{fast.max():.2f}")

print("\nstep-count convergence over all 150 rows (512 vs 2048 substeps):")
for name, x in PARAM_SETS.items():
    coarse = integrate(x, inputs, n_steps=512)
    fine = integrate(x, inputs, n_steps=2048)
    d = np.abs(coarse - fine).max()
    worst_conv = max(worst_conv, d)
    print(f"  {name:11s} max|512-2048| = {d:.3e}")

print(f"\nworst vs BDF:        {worst_bdf:.3e}   (gate < 1e-2)")
print(f"worst step-doubling: {worst_conv:.3e}   (diagnostic; extreme-runaway")
print("                                        corners converge slowest and are")
print("                                        far from where the fit lands --")
print("                                        fit_physics re-asserts this at the")
print("                                        actual fitted parameters)")
if worst_bdf < 0.01:
    print("\nPASS")
else:
    print("\nFAIL — integrator disagrees with the BDF reference")
    sys.exit(1)
