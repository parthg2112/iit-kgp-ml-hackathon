# Provenance of the Submitted Solution

**Team:** Claude ke Chhatore · **Offline round rank:** 13 of 400
**Submission file:** `claude ke chhatore.csv`

This document exists to answer, with evidence, the four requirements in §2 of the Offline
Presentation Round guidelines:

> - The submitted predictions can be generated from the submitted code.
> - The code corresponds to the methodology claimed in the presentation.
> - The methodology presented is consistent with the submitted solution.
> - No post-competition modification is made to improve the leaderboard solution.

---

## 1. The predictions are reproducible from the submitted code

```bash
PY="C:/Users/USER/AppData/Local/Programs/Python/Python311/python.exe"
$PY scripts/make_submission.py
```

This reads the fitted parameters from `artifacts/physics_params.json` and the prediction
policy from `artifacts/blend.json`, and writes `claude ke chhatore.csv`. Regenerating
reproduces the submitted predictions to **4.45e-07** — inside the 6-decimal rounding the file
is written at, i.e. byte-identical output.

**File hashes**

| | |
|---|---|
| sha256 of the file as submitted (first 16) | `c68e0e748e4928f2` |
| sha256 of file content, line endings normalised | `d3411843cd6c0287` |
| Size | 539 bytes · 50 data rows + header |

## 2. The predictions have not changed since before the deadline

The content hash `d3411843cd6c0287` is **identical at every commit from `0b52330` onward**,
which is the commit that adopted the final prediction policy. Verify directly:

```bash
git show 0b52330:claude_ke_chatore.csv | sha256sum   # d3411843cd6c0287
git show HEAD:"claude ke chhatore.csv"  | sha256sum   # d3411843cd6c0287
```

(The file was renamed after the competition to match the registered team name — see §4. The
*contents* are unchanged.)

## 3. Commits after the submission are analysis and documentation only

Every commit listed below post-dates the final predictions. **None changes the model, the
fitted parameters, the prediction policy, or the CSV.** Each is either a measurement whose
result was *rejected*, or a correction to our own documentation.

| Commit | Date | What it is | Changed the predictions? |
|---|---|---|---|
| `0b52330` | 2026-08-14 | **Final policy adopted** — noise-averaged prediction, joint (σ, w, cutoff) = (1.67, 0.87, 60) | **This is the submitted solution** |
| `9c91159` | 2026-08-14 | Documentation: quantified the blend's bias/variance trade; recorded that bagged physics was tested and rejected | No |
| `cc3e2ad` | 2026-08-14 | Re-ran a statistical test on a better-chosen stratum; corrected two stale numbers in docs | No |
| `8e31f43` | 2026-08-14 | Built the pitch package and a consistency-audit script; measured the parallel A→C path (rejected) | No |
| `3da647d` | 2026-08-14 | Presentable HTML deck | No |
| `ccf571f` | 2026-08-14 | Corrected our own arithmetic: per-band RMSE deltas are not additive, redone in SSE | No |
| `2f417f6` | 2026-08-16 | Tested 18 model × feature combinations for a better ensemble member — **rejected**, best gain +0.021 against a 0.3 bar | No |
| `90c5f49` | 2026-08-16 | Corrected two claims in our own docs that did not survive checking (see §5) | No |
| `4be4914` | 2026-08-16 | Renamed the CSV to the registered team name | No — contents identical |

**Nothing in this list was an attempt to improve the leaderboard solution.** The two
experimental commits (`2f417f6`, and the 8-lever sweep documented in `CLAUDE.md`) both
concluded that no change cleared our pre-registered adoption bar, and no change was made.

## 4. The rename

The file was originally written as `claude_ke_chatore.csv`. Two things were wrong with that:
the team name is spelled **Chhatore** (double *h*), and the underscores were our own guess at
what the upload platform preferred rather than anything the rules required. §5 of the problem
statement says the file must be named `[TeamName].csv`, so it now mirrors the registration
exactly. **Contents are byte-for-byte unchanged** — see the hash chain in §2.

## 5. Corrections we made to our own documentation

Listed for completeness, because they change what we *say*, not what we predicted. Both were
found by our own checking after submission:

- **"Leave-one-seed-out prices hyper-parameter selection."** It does not. All CV seeds
  re-partition the same 150 rows, so the held-out seed has already seen every row. Measured at
  row level, selecting the blend weight and cutoff costs **+0.371 RMSE**. Our reported 5.671
  should be read as a lower bound. The *policy* is unaffected — it sits on a plateau, and the
  full-data optimum (0.86, 60) scores 5.6546 against 5.6547 for the shipped (0.87, 60).
- **"Bagging failed because bootstrap replicates fall into a second likelihood basin."** The
  rejection stands, but the mechanism was wrong: basin membership is set by which rows are in
  the fold, and only 7 of 2400 replicates ever crossed.

## 6. What is submitted

| Item | Path |
|---|---|
| Predictions | `claude ke chhatore.csv` |
| Documented notebook (complete workflow) | `notebook/final.ipynb` — self-contained, 46 cells, executes top to bottom |
| Source code | `src/` (model), `scripts/` (every experiment, one question each) |
| Presentation | `docs/pitch/slides.md` · `slides.html` (5 slides) |
| Fitted parameters, CV results, profiles | `artifacts/` |
| Engineering record, including every rejected hypothesis | `CLAUDE.md` |

## 7. Verification you can run

```bash
$PY scripts/check_integrator.py   # the ODE solver agrees with SciPy BDF        -> PASS
$PY scripts/audit_pitch.py        # every number in the pitch vs artifacts/     -> exit 0
$PY scripts/make_submission.py    # regenerates the CSV byte-identically
```

`scripts/audit_pitch.py` mechanically extracts every figure quoted in the presentation and
diffs it against the stored artifacts, verifies the submission's sha256, checks that the
policy triple agrees across all four places it is recorded, and fails if any retracted number
reappears. It is the reason we can claim the presentation matches the code.
