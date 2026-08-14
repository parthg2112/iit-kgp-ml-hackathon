"""Mechanical consistency check: pitch docs vs artifacts vs the shipped CSV.

Why this exists. Twice now a number went stale in one file while staying correct in
another -- the notebook hardcoded sigma = 2.5 after the policy moved to 1.67 and silently
rewrote the submission on re-execution, and a superseded blend weight (0.91) survived in
three places after the joint re-derivation landed on 0.87. Prose review did not catch
either. This does.

The design is registry-driven: every number allowed to appear in the pitch package is
declared in FACTS below, together with the artifact it is read from and the validation
protocol it was measured under. The audit then checks three things:

  1. each fact still matches the artifact it claims to come from
  2. each fact actually appears in the documents that are supposed to state it
  3. no RETRACTED number appears anywhere -- these are claims we made and then falsified,
     and the failure mode is a stale copy resurfacing in a deck

Exit code is non-zero if anything fails, so this can gate a commit.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ROOT
from src.evaluate import BLEND_CUTOFF
from src.physics import NOISE_SIGMA_K

ART = ROOT / "artifacts"
PITCH = ROOT / "docs" / "pitch"
SUBMISSION = ROOT / "claude_ke_chatore.csv"
SUBMISSION_SHA16 = "c68e0e748e4928f2"

DECK, APPENDIX, QA = "deck.md", "appendix.md", "qa_bank.md"


def load(name):
    return json.loads((ART / name).read_text())


def unlabelled_retractions(text, bad):
    """Yield each occurrence of `bad` that is NOT near a superseded-marker."""
    for m in re.finditer(re.escape(bad), text):
        lo = max(0, m.start() - RETRACTION_WINDOW)
        window = text[lo:m.end() + RETRACTION_WINDOW]
        if not RETRACTION_OK.search(window):
            yield m.start()


def norm(text: str) -> str:
    """Prose uses typographic minus/dashes; the formatted facts use ASCII. Normalize so a
    document can read correctly without the audit demanding ASCII hyphens in body text."""
    return (text.replace("−", "-").replace("–", "-").replace("—", "-")
                .replace(" ", " "))


# Some retracted numbers SHOULD appear -- the self-correction section quotes them on
# purpose. What must never happen is one appearing as a live claim. So a retracted value is
# allowed only when the surrounding text marks it as superseded.
RETRACTION_OK = re.compile(
    r"supersed|retract|wrong (in )?stratum|no longer|we (used to|once)|first test|"
    r"falsified|corrected|do not quote|never measured|was wrong|earlier",
    re.I)
RETRACTION_WINDOW = 400


# Numbers that were once claimed and then falsified. If any of these appears WITHOUT a
# superseded-marker nearby, some file is quoting a retracted result as if it were live.
RETRACTED = [
    ("0.615", "corr(tree-physics, y-physics) was never measured at 0.615; the measured "
              "value is in pitch_evidence.json, and the statistic is spurious anyway "
              "(shared -physics term) so it must be labelled as such wherever it appears"),
    ("22 of 42", "superseded sign test -- wrong stratum (mid-range carries none of the "
                 "gain). The decisive test is 8/13 on 0.5 < p <= 10"),
    ("22/42", "superseded sign test -- see above"),
    ("p = 0.878", "0.878 is a blend weight in this repo, never a p-value. The sign-test "
                  "p is 0.88 (mid, superseded) or 0.581 (low, decisive)"),
    ("minutes per run", "the problem statement never quantifies the reference simulation's "
                        "cost; 'minutes' came from our own team brief. Quote the PDF's "
                        "actual wording instead"),
    ("0.91 physics", "superseded policy; shipped weight is 0.87"),
    ("5.711", "superseded sequential blend-search OOF; the shipped figure is LOSO 5.671"),
]

# Deck slide 6 / appendix may legitimately discuss the 10%-RMSE band, but only as a
# labelled display heuristic -- never as the confidence interval.
E2_WIDE_BAND_GUARD = ("210", "320")


def build_facts():
    """Every number the pitch package is allowed to assert, read live from artifacts."""
    params = load("physics_params.json")
    blend = load("blend.json")
    joint = load("joint_policy.json")
    tree = load("tree_sign_test.json")
    ev = load("pitch_evidence.json")
    mc = load("model_comparison.json")
    cold = load("cold_folds.json")
    base = load("baseline_cv.json")
    v = params["vector"]

    F = []

    def add(key, value, fmt, protocol, source, where):
        F.append({"key": key, "value": value, "text": fmt.format(value),
                  "protocol": protocol, "source": source, "where": where})

    # --- kinetics -------------------------------------------------------------
    add("E1", v[1], "{:.2f}", "fitted", "physics_params.json", [DECK, APPENDIX])
    add("E2", v[3], "{:.2f}", "fitted", "physics_params.json", [DECK, APPENDIX])
    add("a1", v[4], "{:.2f}", "fitted", "physics_params.json", [DECK, APPENDIX])
    add("a2", v[5], "{:.2f}", "fitted", "physics_params.json", [DECK, APPENDIX])
    add("U", v[6], "{:.4f}", "fitted", "physics_params.json", [APPENDIX])
    add("E2_over_E1", ev["crossover"]["E2_over_E1"], "{:.2f}", "derived",
        "pitch_evidence.json", [DECK, APPENDIX])
    add("crossover_K", ev["crossover"]["T_K"], "{:.1f}", "derived",
        "pitch_evidence.json", [DECK, APPENDIX])

    # --- the shipped policy ---------------------------------------------------
    add("sigma", blend["sigma"], "{:.2f}", "a priori (EIV median)", "blend.json",
        [DECK, APPENDIX])
    add("weight", blend["weight"], "{:.2f}", "joint OOF search", "blend.json",
        [DECK, APPENDIX])
    add("cutoff", blend["cutoff"], "{:.0f}", "joint OOF search", "blend.json",
        [DECK, APPENDIX])
    add("loso_shipped", joint["loso_mean"]["apriori"], "{:.3f}", "LOSO",
        "joint_policy.json", [DECK, APPENDIX])
    add("loso_previous", joint["loso_mean"]["current"], "{:.3f}", "LOSO",
        "joint_policy.json", [DECK, APPENDIX])

    # --- baselines and protocols ---------------------------------------------
    add("tree_cv", base["ExtraTrees physics"]["mean"], "{:.2f}", "10f-CV",
        "baseline_cv.json", [DECK, APPENDIX])
    add("train_rmse_2048", params["train_rmse_submit_steps"], "{:.4f}",
        "train @2048 substeps", "physics_params.json", [APPENDIX])
    add("neutral_train", mc["neutral"]["train_rmse"], "{:.2f}", "train",
        "model_comparison.json", [DECK, APPENDIX])
    add("cold_mean", cold["held_out_mean"], "{:.2f}", "cold folds",
        "cold_folds.json", [APPENDIX])

    # --- the tree, declared ---------------------------------------------------
    add("sign_low_n", tree["gate"]["n"], "{:.0f}", "OOF", "tree_sign_test.json", [DECK])
    add("sign_low_aligned", tree["gate"]["aligned"], "{:.0f}", "OOF",
        "tree_sign_test.json", [DECK])
    add("sign_low_p", tree["gate"]["p"], "{:.3f}", "OOF", "tree_sign_test.json",
        [DECK, APPENDIX])
    add("corr_err", ev["tree_stats"]["corr_physics_tree_error"], "{:+.3f}", "OOF",
        "pitch_evidence.json", [DECK, APPENDIX])
    add("corr_spurious", ev["tree_stats"]["corr_spurious_shared_physics"], "{:+.3f}",
        "OOF (spurious)", "pitch_evidence.json", [APPENDIX])
    add("tree_standalone", ev["tree_stats"]["tree_standalone_rmse"], "{:.2f}",
        "OOF, blended rows", "pitch_evidence.json", [DECK, APPENDIX])
    add("dead_band_cost", tree["dead"]["band_rmse_gain"], "{:.2f}", "OOF, dead band",
        "tree_sign_test.json", [DECK])

    # --- speed ----------------------------------------------------------------
    add("bdf_speedup", ev["timing"]["speedup_vs_bdf_at_512"], "{:.0f}", "measured",
        "pitch_evidence.json", [DECK, APPENDIX])

    # --- falsifications -------------------------------------------------------
    if ev.get("parallel_AC", {}).get("verified"):
        add("ac_cv_delta", ev["parallel_AC"]["cv_delta"], "{:+.3f}", "10f-CV, 3 seeds",
            "pitch_evidence.json", [DECK, APPENDIX])
    return F


def check_docs(facts, problems, notes):
    texts = {}
    for name in (DECK, APPENDIX, QA):
        p = PITCH / name
        if not p.exists():
            problems.append(f"MISSING DOCUMENT: docs/pitch/{name}")
            texts[name] = ""
        else:
            texts[name] = norm(p.read_text(encoding="utf-8"))

    for f in facts:
        for doc in f["where"]:
            if texts.get(doc) and f["text"] not in texts[doc]:
                problems.append(
                    f"{doc}: fact '{f['key']}' = {f['text']} ({f['protocol']}, "
                    f"from {f['source']}) does not appear")

    for name, text in texts.items():
        for bad, why in RETRACTED:
            for pos in unlabelled_retractions(text, bad):
                problems.append(
                    f"{name}: RETRACTED value '{bad}' at char {pos} is not marked as "
                    f"superseded -- {why}")

    # The wide E2 band may appear only if explicitly labelled as the display heuristic.
    for name, text in texts.items():
        for line in text.splitlines():
            if all(t in line for t in E2_WIDE_BAND_GUARD) and "E" in line:
                if not re.search(r"10\s*%|heuristic|display|flat bottom|basin", line, re.I):
                    problems.append(
                        f"{name}: the wide E2 band [210, 320] appears unlabelled -- it is a "
                        f"display heuristic, not the confidence interval. Line: {line[:90]}")
    return texts


def check_shipped(problems, notes):
    if not SUBMISSION.exists():
        problems.append("submission CSV missing")
        return
    sha = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()[:16]
    if sha != SUBMISSION_SHA16:
        problems.append(f"submission sha256 is {sha}, expected {SUBMISSION_SHA16}")
    else:
        notes.append(f"submission sha256 {sha} OK")

    rows = [r for r in SUBMISSION.read_text().strip().splitlines()]
    if rows[0].strip() != "overall_yield":
        problems.append(f"submission header is {rows[0]!r}, expected 'overall_yield'")
    if len(rows) - 1 != 50:
        problems.append(f"submission has {len(rows) - 1} data rows, expected 50")
    vals = [float(r) for r in rows[1:]]
    if not all(0.0 <= v <= 100.0 for v in vals):
        problems.append("submission has values outside [0, 100]")
    notes.append(f"submission {len(vals)} rows, range "
                 f"[{min(vals):.3f}, {max(vals):.3f}] OK")

    # The policy triple must agree across every place it is recorded.
    blend = load("blend.json")
    triple = {
        "blend.json": (blend["sigma"], blend["weight"], blend["cutoff"]),
        "src/physics.py + src/evaluate.py": (NOISE_SIGMA_K, blend["weight"], BLEND_CUTOFF),
    }
    if len(set(triple.values())) != 1:
        problems.append(f"policy triple disagrees across sources: {triple}")
    else:
        notes.append(f"policy triple (sigma, w, cutoff) = "
                     f"{triple['blend.json']} consistent")


def check_notebook(problems, notes):
    nb_path = ROOT / "notebook" / "final.ipynb"
    if not nb_path.exists():
        problems.append("notebook/final.ipynb missing")
        return
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    errs = [c for c in nb["cells"]
            if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    if errs:
        problems.append(f"notebook has {len(errs)} cells with error output")
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    if re.search(r"^\s*(from|import)\s+src\b", src, re.M):
        problems.append("notebook imports from src/ -- it must be self-contained")
    src = norm(src)
    for bad, why in RETRACTED:
        for pos in unlabelled_retractions(src, bad):
            problems.append(
                f"notebook: RETRACTED value '{bad}' at char {pos} is not marked as "
                f"superseded -- {why}")
    notes.append(f"notebook {len(nb['cells'])} cells, 0 errors, self-contained OK")


def main() -> int:
    problems, notes = [], []
    facts = build_facts()
    check_shipped(problems, notes)
    check_notebook(problems, notes)
    check_docs(facts, problems, notes)

    print("=" * 74)
    print("PITCH CONSISTENCY AUDIT")
    print("=" * 74)
    print(f"\n{len(facts)} declared facts, each read live from its artifact:\n")
    print(f"  {'fact':18s} {'value':>10s}  {'protocol':<24s} source")
    for f in facts:
        print(f"  {f['key']:18s} {f['text']:>10s}  {f['protocol']:<24s} {f['source']}")

    print("\nchecks passed:")
    for n in notes:
        print(f"  OK  {n}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  !!  {p}")
        print("\nAUDIT FAILED")
        return 1
    print("\nno inconsistencies found. AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
