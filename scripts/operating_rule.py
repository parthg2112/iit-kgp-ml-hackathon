"""What the 449.9 K crossover does and does not license us to say.

The crossover is where the RATE CONSTANTS cross: above it k2 > k1, so the waste reaction is
intrinsically faster than the desired one. It is tempting to close a pitch on "keep the
reactor below 449.9 K or selectivity collapses" -- and that statement is falsifiable from
our own data, because yield depends on k*tau, not on k alone. Run hot but exit fast and B
leaves before it degrades.

This script measures the counterexamples so the deck states a joint (T, tau) condition
instead of a temperature ceiling.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT, TARGET, load_test, load_train
from src.physics import NOISE_SIGMA_K, SUBMIT_STEPS, _rate, predict_smoothed

ARTIFACTS = ROOT / "artifacts"
T_CROSS = 449.894


def main() -> int:
    x = json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"]
    tr, te = load_train(), load_test()

    def geom(df):
        return ((df.inlet_temperature_K + df.jacket_temperature_K) / 2).to_numpy(float), \
               (df.length_m / df.flow_rate_L_min).to_numpy(float)

    T_tr, tau_tr = geom(tr)
    T_te, tau_te = geom(te)
    y = tr[TARGET].to_numpy(float)
    p_te = predict_smoothed(x, te, NOISE_SIGMA_K, n_steps=SUBMIT_STEPS)

    hot_tr = T_tr > T_CROSS
    hot_te = T_te > T_CROSS

    print(f"crossover k1 = k2 at {T_CROSS:.1f} K\n")
    print("IS IT AN OPERATING CEILING? -- observed training evidence above the crossover")
    print(f"  rows with T_avg > {T_CROSS:.1f} K: {int(hot_tr.sum())} of {len(tr)}")
    print(f"  MAX OBSERVED YIELD among them: {y[hot_tr].max():.2f}%   "
          f"<- a ceiling would forbid this")
    print(f"  their tau range: {tau_tr[hot_tr].min():.4f} - {tau_tr[hot_tr].max():.4f} "
          f"(all rows: {tau_tr.min():.4f} - {tau_tr.max():.4f})")
    n_good = int(((y > 40) & hot_tr).sum())
    print(f"  rows above the crossover that still yield > 40%: {n_good}")

    # The best hot row, and what makes it work.
    best = int(np.argmax(np.where(hot_tr, y, -1)))
    k1b = float(_rate(x[0], x[1], np.array([T_tr[best]]))[0])
    k2b = float(_rate(x[2], x[3], np.array([T_tr[best]]))[0])
    print(f"\n  best hot training row: T_avg {T_tr[best]:.1f} K, tau {tau_tr[best]:.4f}, "
          f"yield {y[best]:.2f}%")
    print(f"    k1*tau = {k1b*tau_tr[best]:.2f}   k2*tau = {k2b*tau_tr[best]:.2f}"
          f"   -> A converts, B exits before it degrades")

    print("\nTEST rows above the crossover that we predict will still perform")
    print(f"  {'T_avg':>8}{'tau':>9}{'k1*tau':>9}{'k2*tau':>10}{'predicted':>11}")
    rows = []
    for i in np.argsort(-np.where(hot_te, p_te, -1))[:5]:
        if not hot_te[i]:
            continue
        k1 = float(_rate(x[0], x[1], np.array([T_te[i]]))[0])
        k2 = float(_rate(x[2], x[3], np.array([T_te[i]]))[0])
        print(f"  {T_te[i]:>8.1f}{tau_te[i]:>9.4f}{k1*tau_te[i]:>9.2f}"
              f"{k2*tau_te[i]:>10.2f}{p_te[i]:>11.2f}")
        rows.append({"T_avg": float(T_te[i]), "tau": float(tau_te[i]),
                     "k1_tau": float(k1 * tau_te[i]), "k2_tau": float(k2 * tau_te[i]),
                     "predicted": float(p_te[i])})

    print("\nCONCLUSION for the deck:")
    print("  449.9 K is NOT a ceiling on temperature. It is the point beyond which residence")
    print("  time must be actively shortened -- above it every extra unit of tau costs")
    print("  selectivity, because k2 now outruns k1. The operating constraint is joint in")
    print("  (T, tau), and stating it as a temperature limit alone is refutable from our own")
    print("  training data.")

    out = {"T_cross_K": T_CROSS,
           "train_rows_above": int(hot_tr.sum()),
           "max_observed_yield_above": float(y[hot_tr].max()),
           "train_above_yielding_over_40": n_good,
           "tau_range_above": [float(tau_tr[hot_tr].min()), float(tau_tr[hot_tr].max())],
           "tau_range_all": [float(tau_tr.min()), float(tau_tr.max())],
           "best_hot_train": {"T_avg": float(T_tr[best]), "tau": float(tau_tr[best]),
                              "yield": float(y[best]),
                              "k1_tau": float(k1b * tau_tr[best]),
                              "k2_tau": float(k2b * tau_tr[best])},
           "test_rows_above": int(hot_te.sum()),
           "top_hot_test_rows": rows,
           "verdict": "joint (T, tau) condition, not a temperature ceiling"}
    (ARTIFACTS / "operating_rule.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ARTIFACTS / 'operating_rule.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
