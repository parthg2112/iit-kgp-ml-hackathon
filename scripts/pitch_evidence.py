"""Close every evidence gap the pitch package needs, and write them to one artifact.

Each block here exists because a claim we intended to make had nothing measured behind it.
The rule for the deck is "no claim without a measurement", so anything this script cannot
produce a number for does not go on a slide.

1. PARALLEL A->C PATH. The problem statement calls the network "series-parallel" but lists
   only A->B->C. Our rebuttal existed only as prose. This fits the parallel path properly.
   The analytic step generalises exactly: A decays at kA = k1 + k3 while B is produced at
   k1 only, so the scheme stays unconditionally stable -- no fallback solver, no fixed-step
   RK4. Verified against scipy BDF before any number is trusted.
   src/physics.py is deliberately NOT modified: the model is frozen, and check_integrator.py
   must keep passing against the shipped core.
2. TIMING. "~22 ms" was asserted in three files and never measured. Measured here, against
   a scipy BDF reference solve, so the ratio is measured end to end.
3. F-TEST INTERVALS as stored scalars. They lived only in notebook output and prose.
4. TREE/PHYSICS error statistics, including the spurious one we tell people not to quote --
   computed once so the disclosure names a real number instead of an invented one.
5. CROSSOVER TEMPERATURE from the fitted parameters, with the training envelope beside it.
"""

import json
import sys
import time
from multiprocessing import Pool, freeze_support
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import f as f_dist
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import R_GAS, ROOT, T_REF, TARGET, load_train, ode_inputs, rmse
from src.evaluate import BLEND_CUTOFF, apply_blend
from src.physics import (DEFAULT_STEPS, LOWER, NOISE_SIGMA_K, SUBMIT_STEPS, UPPER,
                         _rate, _series_step, _temperature_step, integrate,
                         reference_solve, residuals)

ARTIFACTS = ROOT / "artifacts"
N_TRAIN, N_PARAM = 150, 7

# --- A->C variant -----------------------------------------------------------------
# (ln_k1_ref, E1_kJ, ln_k2_ref, E2_kJ, a1, a2, U, ln_k3_ref, E3_kJ, a3); n_flow pinned 0.
AC_NAMES = ("ln_k1_ref", "E1_kJ", "ln_k2_ref", "E2_kJ", "a1", "a2", "U",
            "ln_k3_ref", "E3_kJ", "a3")
AC_LOWER = np.array([-12.0, 40.0, -12.0, 60.0, -30.0, -30.0, 0.0, -12.0, 40.0, -30.0])
AC_UPPER = np.array([12.0, 140.0, 12.0, 280.0, 30.0, 30.0, 60.0, 12.0, 280.0, 30.0])
AC_POISON = 1e3


def ac_integrate(x, inputs, n_steps=DEFAULT_STEPS):
    """Same splitting scheme as src.physics.integrate, with a parallel A->C path.

        dxA/dz = -(k1 + k3) xA
        dxB/dz =  k1 xA - k2 xB

    A decays at kA = k1 + k3 but B is produced at k1 only, so `_series_step` is reused with
    kA for the decay and the gain rescaled by k1/kA. At k3 = 0 this reduces to the shipped
    model exactly -- which is the first thing the verification below checks.
    """
    (ln_k1, E1, ln_k2, E2, a1, a2, U, ln_k3, E3, a3) = [float(v) for v in x]
    CA0, T_jacket, tau = inputs["CA0"], inputs["T_jacket"], inputs["tau"]
    dz = tau / n_steps
    half = 0.5 * dz

    xA = np.ones_like(CA0)
    xB = np.zeros_like(CA0)
    T = inputs["T_in"].copy()

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(n_steps):
            for step, dzz in ((0, half), (1, dz)):
                Tuse = T if step == 0 else T_mid
                k1 = _rate(ln_k1, E1, Tuse)
                k2 = _rate(ln_k2, E2, Tuse)
                k3 = _rate(ln_k3, E3, Tuse)
                kA = k1 + k3
                # `_series_step` returns gain built from kA; rescale to the k1 branch.
                xA_n, xB_g = _series_step(xA, xB, kA, k2, dzz)
                frac1 = np.where(kA > 0, k1 / np.where(kA > 0, kA, 1.0), 1.0)
                xB_n = xB * np.exp(-k2 * dzz) + (xB_g - xB * np.exp(-k2 * dzz)) * frac1

                consumed_A = xA - xA_n
                to_B = consumed_A * frac1
                to_C_direct = consumed_A - to_B
                consumed_B = to_B - (xB_n - xB)
                heat = CA0 * (a1 * to_B + a2 * consumed_B + a3 * to_C_direct)
                if step == 0:
                    T_mid = _temperature_step(T, T_jacket, heat, U, dzz)
                else:
                    T = _temperature_step(T, T_jacket, heat, U, dzz)
                    xA, xB = xA_n, xB_n

    y = 100.0 * xB
    return np.where(np.isfinite(y), np.clip(y, 0.0, 100.0), 0.0)


def ac_reference(x, inputs, row):
    """Independent scipy BDF solve of the A->C system for one row."""
    (ln_k1, E1, ln_k2, E2, a1, a2, U, ln_k3, E3, a3) = [float(v) for v in x]
    CA0 = float(inputs["CA0"][row])
    Tj = float(inputs["T_jacket"][row])
    tau = float(inputs["tau"][row])

    def rhs(_z, s):
        xA, xB, T = s
        k1 = np.exp(ln_k1 - (E1 * 1000.0 / R_GAS) * (1.0 / T - 1.0 / T_REF))
        k2 = np.exp(ln_k2 - (E2 * 1000.0 / R_GAS) * (1.0 / T - 1.0 / T_REF))
        k3 = np.exp(ln_k3 - (E3 * 1000.0 / R_GAS) * (1.0 / T - 1.0 / T_REF))
        dT = CA0 * (a1 * k1 * xA + a2 * k2 * xB + a3 * k3 * xA) + U * (Tj - T)
        return [-(k1 + k3) * xA, k1 * xA - k2 * xB, dT]

    sol = solve_ivp(rhs, (0.0, tau), [1.0, 0.0, float(inputs["T_in"][row])],
                    method="BDF", rtol=1e-10, atol=1e-12, dense_output=True)
    return 100.0 * float(sol.y[1, -1])


def ac_residuals(x, inputs, y, n_steps, convergence_tol):
    pred = ac_integrate(x, inputs, n_steps=n_steps)
    if convergence_tol is not None:
        coarse = ac_integrate(x, inputs, n_steps=max(8, n_steps // 2))
        if np.abs(pred - coarse).max() > convergence_tol:
            return np.full_like(y, AC_POISON, dtype=float)
    r = pred - y
    return np.where(np.isfinite(r), r, AC_POISON)


def ac_objective(x, inputs, y, n_steps, convergence_tol):
    """Module-level so differential_evolution can pickle it and actually use workers=-1."""
    r = ac_residuals(x, inputs, y, n_steps, convergence_tol)
    return float(np.sqrt(np.mean(r ** 2)))


def _fold_job(args):
    """One (model, seed, fold) refit. Run in a Pool: parallelism one level up, each fit
    single-threaded, per the note in CLAUDE.md about not oversubscribing the cores."""
    kind, seed, tr_idx, te_idx, x0 = args
    tr = load_train()
    sub = tr.iloc[tr_idx]
    inputs, y = ode_inputs(sub), sub[TARGET].to_numpy(float)
    fn = ac_residuals if kind == "ac" else residuals
    lo, up = (AC_LOWER, AC_UPPER) if kind == "ac" else (LOWER, UPPER)
    r = least_squares(fn, x0=np.asarray(x0, float), bounds=(lo, up),
                      args=(inputs, y, DEFAULT_STEPS, None),
                      x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
    held = tr.iloc[te_idx]
    pred = (ac_integrate if kind == "ac" else integrate)(r.x, ode_inputs(held))
    return kind, seed, list(te_idx), list(pred)


def fit_ac(inputs, y, seed, maxiter=180):
    de = differential_evolution(
        ac_objective, bounds=list(zip(AC_LOWER, AC_UPPER)),
        args=(inputs, y, 256, 0.05),          # guard ON here only -- a DE call site
        strategy="best1bin", maxiter=maxiter, popsize=18, tol=1e-8,
        mutation=(0.4, 1.0), recombination=0.85, seed=seed,
        polish=False, workers=-1, updating="deferred", init="sobol")
    r = least_squares(ac_residuals, x0=de.x, bounds=(AC_LOWER, AC_UPPER),
                      args=(inputs, y, DEFAULT_STEPS, None),   # guard OFF for the polish
                      x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
    return r.x, float(np.sqrt(np.mean(ac_residuals(
        r.x, inputs, y, SUBMIT_STEPS, None) ** 2)))


def ftest_interval(points, ref_rmse, alpha):
    """Admissible interval by the likelihood-ratio (F) criterion, linearly interpolated
    between grid points. This is the statistically meaningful band -- NOT the 10%-RMSE
    display heuristic, which is far too generous."""
    thr = ref_rmse * np.sqrt(1.0 + f_dist.ppf(alpha, 1, N_TRAIN - N_PARAM) / (N_TRAIN - N_PARAM))
    xs = np.array(sorted(float(k) for k in points))
    rs = np.array([points[k]["train_rmse"] for k in sorted(points, key=float)])
    best = xs[int(np.argmin(rs))]

    def cross(lo_side):
        idx = range(int(np.argmin(rs)), 0, -1) if lo_side else range(int(np.argmin(rs)), len(xs) - 1)
        for i in idx:
            j = i - 1 if lo_side else i + 1
            if rs[j] >= thr:
                t = (thr - rs[i]) / (rs[j] - rs[i]) if rs[j] != rs[i] else 0.0
                return float(xs[i] + t * (xs[j] - xs[i]))
        return float(xs[0] if lo_side else xs[-1])

    return {"best": float(best), "threshold_rmse": float(thr),
            "lo": cross(True), "hi": cross(False)}


def main() -> int:
    tr = load_train()
    y = tr[TARGET].to_numpy(dtype=float)
    inputs = ode_inputs(tr)
    fitted = np.array(json.loads((ARTIFACTS / "physics_params.json").read_text())["vector"])
    out = {}

    # ---------------------------------------------------------------- crossover T
    ln_k1, E1, ln_k2, E2, a1, a2, U = fitted[:7]
    inv_T = 1.0 / T_REF + (R_GAS / 1000.0) * (ln_k1 - ln_k2) / (E1 - E2)
    T_cross = 1.0 / inv_T
    T_all = np.concatenate([tr["inlet_temperature_K"], tr["jacket_temperature_K"]])
    print(f"crossover k1 == k2 at T = {T_cross:.1f} K   "
          f"(training temperatures span {T_all.min():.1f}-{T_all.max():.1f} K, "
          f"inside = {T_all.min() < T_cross < T_all.max()})")
    print(f"E2/E1 = {E2 / E1:.3f}")
    out["crossover"] = {"T_K": float(T_cross), "T_min": float(T_all.min()),
                        "T_max": float(T_all.max()), "E2_over_E1": float(E2 / E1),
                        "inside_envelope": bool(T_all.min() < T_cross < T_all.max())}

    # ---------------------------------------------------------------- timing
    print("\ntiming (measured, not asserted):")
    timing = {}
    for steps in (256, DEFAULT_STEPS, SUBMIT_STEPS):
        integrate(fitted, inputs, n_steps=steps)          # warm up
        reps = 20 if steps <= 512 else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            integrate(fitted, inputs, n_steps=steps)
        ms = (time.perf_counter() - t0) / reps * 1000.0
        timing[f"integrate_{steps}_ms_150rows"] = ms
        print(f"  our integrator, {steps:5d} substeps, 150 rows: {ms:8.2f} ms "
              f"({ms / len(tr) * 1000:.1f} us/row)")
    idx = list(range(len(tr)))
    t0 = time.perf_counter()
    ref = reference_solve(fitted, inputs, idx)
    ref_ms = (time.perf_counter() - t0) * 1000.0
    timing["reference_solve_ms_150rows"] = ref_ms
    timing["speedup_vs_bdf_at_512"] = ref_ms / timing[f"integrate_{DEFAULT_STEPS}_ms_150rows"]
    timing["speedup_vs_bdf_at_2048"] = ref_ms / timing[f"integrate_{SUBMIT_STEPS}_ms_150rows"]
    agree = float(np.abs(integrate(fitted, inputs, n_steps=SUBMIT_STEPS) - ref).max())
    timing["max_abs_diff_vs_bdf"] = agree
    print(f"  scipy BDF reference solve, 150 rows:            {ref_ms:8.2f} ms")
    print(f"  -> {timing['speedup_vs_bdf_at_512']:.0f}x faster than BDF at 512 substeps, "
          f"{timing['speedup_vs_bdf_at_2048']:.0f}x at {SUBMIT_STEPS}")
    print(f"  and they agree: max |ours - BDF| = {agree:.2e} yield-points")
    out["timing"] = timing

    # ---------------------------------------------------------------- F-test intervals
    print("\nF-test admissible intervals (the quoted band; NOT the 10%-RMSE heuristic):")
    out["ftest"] = {}
    for name, fname in (("E2_kJ", "profile_E2_kJ"), ("a1", "profile_a1"),
                        ("n_flow", "profile_n_flow")):
        prof = json.loads((ARTIFACTS / f"{fname}.json").read_text())
        rec = {"one_sigma": ftest_interval(prof["points"], prof["reference_rmse"], 0.6827),
               "ninety_five": ftest_interval(prof["points"], prof["reference_rmse"], 0.95)}
        out["ftest"][name] = rec
        print(f"  {name:8s} best {rec['ninety_five']['best']:8.2f}   "
              f"1s [{rec['one_sigma']['lo']:.2f}, {rec['one_sigma']['hi']:.2f}]   "
              f"95% [{rec['ninety_five']['lo']:.2f}, {rec['ninety_five']['hi']:.2f}]")

    # ---------------------------------------------------------------- tree statistics
    z = np.load(ARTIFACTS / "oof_by_sigma.npz")
    phys_seeds = z[f"s_{NOISE_SIGMA_K}"]
    tree_seeds = np.load(ARTIFACTS / "tree_oof.npy")
    w = json.loads((ARTIFACTS / "blend.json").read_text())["weight"]
    phys, tree = phys_seeds.mean(0), tree_seeds.mean(0)
    m = phys <= BLEND_CUTOFF                       # the rows the blend actually touches
    e_p, e_t = phys[m] - y[m], tree[m] - y[m]
    honest = float(np.corrcoef(e_p, e_t)[0, 1])
    spurious = float(np.corrcoef(tree[m] - phys[m], y[m] - phys[m])[0, 1])
    blended = apply_blend(phys, tree, w, BLEND_CUTOFF)
    print("\ntree/physics error statistics on the blended rows "
          f"(n={int(m.sum())}, seed-averaged OOF):")
    print(f"  corr(physics error, tree error)      {honest:+.3f}   <- the honest statistic")
    print(f"  corr(tree-physics, y-physics)        {spurious:+.3f}   <- SPURIOUS, shares -physics")
    print(f"  tree standalone RMSE                 {rmse(y[m], tree[m]):.3f}")
    print(f"  physics RMSE                         {rmse(y[m], phys[m]):.3f}")
    print(f"  blended RMSE                         {rmse(y[m], blended[m]):.3f}")
    print(f"  mean signed residual  physics {np.mean(y[m] - phys[m]) * -1:+.3f}"
          f" -> blended {np.mean(blended[m] - y[m]):+.3f}")
    out["tree_stats"] = {
        "n_blended_rows": int(m.sum()), "corr_physics_tree_error": honest,
        "corr_spurious_shared_physics": spurious,
        "tree_standalone_rmse": float(rmse(y[m], tree[m])),
        "physics_rmse": float(rmse(y[m], phys[m])),
        "blended_rmse": float(rmse(y[m], blended[m])),
        "mean_signed_physics": float(np.mean(phys[m] - y[m])),
        "mean_signed_blended": float(np.mean(blended[m] - y[m])),
        "weight": w}

    # ---------------------------------------------------------------- parallel A->C
    print("\n" + "=" * 78)
    print("PARALLEL A->C PATH -- the problem statement says 'series-parallel'")
    print("=" * 78)

    # Reduction check: at k3 -> 0 the A->C scheme must reproduce the shipped model exactly.
    # ln_k3 = -40 puts k3 at ~4e-18 (below the fit bound of -12, but this is a call, not a
    # fit -- at the bound itself k3 ~ 6e-6 is small yet not numerically zero).
    x_ac0 = np.concatenate([fitted[:7], [-40.0, 100.0, 0.0]])
    red = np.abs(ac_integrate(x_ac0, inputs) - integrate(fitted, inputs)).max()
    print(f"reduction check: |A->C at k3->0  -  shipped model| = {red:.2e} yield-points")

    x_probe = np.concatenate([fitted[:7], [1.0, 90.0, -5.0]])   # k3 genuinely active
    rows = [int(i) for i in np.argsort(-inputs["tau"])[:3]] + [0, 74]
    dev = max(abs(ac_integrate(x_probe, inputs)[i] - ac_reference(x_probe, inputs, i))
              for i in rows)
    print(f"BDF check with k3 ACTIVE: max |ours - scipy BDF| = {dev:.2e} yield-points "
          f"over rows {rows}")
    if not (red < 1e-6 and dev < 1e-2):
        print("!! A->C integrator failed verification; refusing to report a fit")
        out["parallel_AC"] = {"verified": False, "reduction_err": float(red),
                              "bdf_err": float(dev)}
    else:
        best_x, best_rmse = None, np.inf
        for s in (11, 12):
            xs, rs = fit_ac(inputs, y, seed=s)
            print(f"  A->C fit seed {s}: train RMSE {rs:.4f}  "
                  f"ln_k3_ref {xs[7]:+.3f} (bound {AC_LOWER[7]:+.1f})")
            if rs < best_rmse:
                best_x, best_rmse = xs, rs
        series_train = float(np.sqrt(np.mean(
            residuals(fitted, inputs, y, SUBMIT_STEPS, None) ** 2)))
        at_bound = bool(best_x[7] <= AC_LOWER[7] + 1e-3)
        shift = {n: float(best_x[i] - fitted[i]) for i, n in enumerate(AC_NAMES[:7])}
        print(f"\n  series train RMSE {series_train:.4f} (7 free, {SUBMIT_STEPS} substeps)")
        print(f"  A->C   train RMSE {best_rmse:.4f} (10 free)   "
              f"delta {series_train - best_rmse:+.4f}")
        print(f"  ln_k3_ref at lower bound: {at_bound}")
        print("  shift in the other seven parameters: " +
              ", ".join(f"{k} {v:+.2e}" for k, v in shift.items()))

        print("\n  honest gate: 10-fold CV, 3 seeds, both models refit per fold")
        jobs = []
        for seed in (0, 1, 2):
            kf = KFold(n_splits=10, shuffle=True, random_state=seed)
            for tri, tei in kf.split(tr):
                jobs.append(("series", seed, tri, tei, fitted))
                jobs.append(("ac", seed, tri, tei, best_x))
        with Pool(processes=10) as pool:
            results = pool.map(_fold_job, jobs)
        oof = {("series", s): np.zeros(len(tr)) for s in (0, 1, 2)}
        oof.update({("ac", s): np.zeros(len(tr)) for s in (0, 1, 2)})
        for kind, seed, tei, pred in results:
            oof[(kind, seed)][np.array(tei)] = np.array(pred)
        cv = {k: float(rmse(y, v)) for k, v in oof.items()}
        for s in (0, 1, 2):
            print(f"    seed {s}: series {cv[('series', s)]:.4f}   "
                  f"A->C {cv[('ac', s)]:.4f}   delta {cv[('series', s)] - cv[('ac', s)]:+.4f}")
        ms = float(np.mean([cv[("series", s)] for s in (0, 1, 2)]))
        ma = float(np.mean([cv[("ac", s)] for s in (0, 1, 2)]))
        print(f"    mean:   series {ms:.4f}   A->C {ma:.4f}   delta {ms - ma:+.4f}"
              f"   (bar was +0.3)")
        print(f"  VERDICT: {'ADOPT' if ms - ma > 0.3 else 'REJECT -- series stands'}")
        out["parallel_AC"] = {
            "verified": True, "reduction_err": float(red), "bdf_err": float(dev),
            "vector": [float(v) for v in best_x], "names": list(AC_NAMES),
            "ln_k3_at_lower_bound": at_bound, "lower_bound_ln_k3": float(AC_LOWER[7]),
            "train_rmse_series": series_train, "train_rmse_ac": float(best_rmse),
            "param_shift": shift, "cv_series": ms, "cv_ac": ma,
            "cv_delta": float(ms - ma), "bar": 0.3,
            "cv_per_seed": {f"{k[0]}_{k[1]}": v for k, v in cv.items()},
            "verdict": "ADOPT" if ms - ma > 0.3 else "REJECT"}

    (ARTIFACTS / "pitch_evidence.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ARTIFACTS / 'pitch_evidence.json'}")
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
