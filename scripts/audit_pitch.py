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
    # NOTE: tree_sign_test.json also carries a dead-band cost (-1.55) computed on
    # seed-AVERAGED predictions. It is deliberately NOT a declared fact: quoting both it and
    # the pooled -1.594 puts two different numbers for one quantity in front of a judge.
    # The pooled figure below is the one the docs state, because it is the basis whose band
    # contributions sum exactly.

    # --- reconciled band accounting (SSE, which is additive) --------------------
    acc = load("blend_accounting.json")
    by = {b["band"]: b for b in acc["bands"]}
    add("sse_total_gain", acc["total_sse_delta"], "{:.1f}", "10f-CV pooled, 3 seeds",
        "blend_accounting.json", [DECK, APPENDIX])
    add("sse_low", by["low"]["sse_delta"], "{:+.1f}", "10f-CV pooled", "blend_accounting.json",
        [DECK, APPENDIX])
    add("sse_mid", by["mid"]["sse_delta"], "{:+.1f}", "10f-CV pooled", "blend_accounting.json",
        [DECK, APPENDIX])
    add("sse_dead", by["dead"]["sse_delta"], "{:.1f}", "10f-CV pooled", "blend_accounting.json",
        [DECK, APPENDIX])
    add("dead_rmse_pre", by["dead"]["rmse_pre"], "{:.3f}", "10f-CV pooled, dead band",
        "blend_accounting.json", [DECK, APPENDIX])
    add("dead_rmse_post", by["dead"]["rmse_post"], "{:.3f}", "10f-CV pooled, dead band",
        "blend_accounting.json", [DECK, APPENDIX])
    add("dead_rmse_delta", by["dead"]["rmse_delta"], "{:.3f}", "10f-CV pooled, dead band",
        "blend_accounting.json", [DECK, APPENDIX])

    # --- the operating rule is a JOINT (T, tau) condition, not a ceiling --------
    op = load("operating_rule.json")
    add("max_yield_above_crossover", op["max_observed_yield_above"], "{:.0f}", "observed",
        "operating_rule.json", [DECK, APPENDIX])
    add("train_rows_above_crossover", op["train_rows_above"], "{:.0f}", "observed",
        "operating_rule.json", [DECK])

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


def check_arithmetic(problems, notes):
    """A different defect class from a wrong number: every figure individually correct, the
    SET collectively unreconcilable. That is what happened when the deck asserted the whole
    blend gain came from 13 rows while also reporting a loss on the near-zero band -- both
    true, jointly impossible, because per-band RMSE deltas are not additive.

    So: assert that band contributions SUM, that every quoted delta equals the difference of
    its quoted levels, and that every quoted ratio equals its quoted operands.
    """
    acc = load("blend_accounting.json")
    ev = load("pitch_evidence.json")
    params = load("physics_params.json")
    v = params["vector"]

    # (a) band-level SSE contributions must sum to the total
    band_sum = sum(b["sse_delta"] for b in acc["bands"])
    total = acc["total_sse_delta"]
    if abs(band_sum - total) > 1e-6:
        problems.append(f"band SSE deltas sum to {band_sum:.4f}, total is {total:.4f}")
    else:
        notes.append(f"band SSE contributions sum to the total "
                     f"({total:.1f}, residual {abs(band_sum-total):.1e})")

    # total SSE delta must equal pre minus post
    if abs((acc["total_sse_pre"] - acc["total_sse_post"]) - total) > 1e-6:
        problems.append("total_sse_delta != total_sse_pre - total_sse_post")

    # (b) every band's quoted RMSE delta must equal the difference of its quoted levels,
    #     and its RMSE levels must be consistent with its own SSE and n
    for b in acc["bands"]:
        if abs((b["rmse_pre"] - b["rmse_post"]) - b["rmse_delta"]) > 5e-4:
            problems.append(f"band {b['band']}: rmse_delta {b['rmse_delta']:.4f} != "
                            f"{b['rmse_pre']:.4f} - {b['rmse_post']:.4f}")
        if b["n"]:
            for tag in ("pre", "post"):
                implied = (b[f"sse_{tag}"] / b["n"]) ** 0.5
                if abs(implied - b[f"rmse_{tag}"]) > 5e-4:
                    problems.append(
                        f"band {b['band']}: rmse_{tag} {b[f'rmse_{tag}']:.4f} inconsistent "
                        f"with sse_{tag}/n -> {implied:.4f}")

    # pooled RMSE must be consistent with pooled SSE and N
    N = acc["basis"]["n_predictions"]
    for tag in ("pre", "post"):
        implied = (acc[f"total_sse_{tag}"] / N) ** 0.5
        if abs(implied - acc["basis"][f"pooled_{tag}"]) > 5e-4:
            problems.append(f"pooled_{tag} {acc['basis'][f'pooled_{tag}']:.4f} inconsistent "
                            f"with sqrt(SSE/N) = {implied:.4f}")
    notes.append("every band RMSE level reconciles with its own SSE and n")

    # (c) quoted ratios must equal their quoted operands
    ratio = ev["crossover"]["E2_over_E1"]
    if abs(ratio - v[3] / v[1]) > 1e-6:
        problems.append(f"E2/E1 stated as {ratio:.4f} but E2/E1 = {v[3]/v[1]:.4f}")
    speed = ev["timing"]["speedup_vs_bdf_at_512"]
    implied = ev["timing"]["reference_solve_ms_150rows"] / ev["timing"]["integrate_512_ms_150rows"]
    if abs(speed - implied) > 1e-6:
        problems.append(f"BDF speed-up stated as {speed:.2f} but timings imply {implied:.2f}")
    notes.append("quoted ratios (E2/E1, BDF speed-up) match their operands")

    # the LOSO gain must equal the difference of the two LOSO levels
    joint = load("joint_policy.json")["loso_mean"]
    gain = joint["current"] - joint["apriori"]
    if abs(gain - 0.3511) > 5e-4:
        problems.append(f"LOSO gain {gain:.4f} no longer matches the documented +0.3511")
    notes.append(f"LOSO gain {joint['current']:.3f} - {joint['apriori']:.3f} = {gain:.4f}")


def check_deck_parity(problems, notes):
    """deck.md and deck.html are two renderings of one deck, edited by hand. A number fixed
    in one and missed in the other is invisible in review, so compare their numeric literals
    directly. CSS lengths and the in-page crossover computation are excluded -- the latter
    deliberately recomputes 449.9 K from the fitted parameters so the figure cannot drift."""
    md_p, html_p = PITCH / DECK, PITCH / "deck.html"
    if not (md_p.exists() and html_p.exists()):
        return

    def literals(text, strip_markup=False):
        text = norm(text)
        if strip_markup:
            text = re.sub(r"<style>.*?</style>", " ", text, flags=re.S)
            text = re.sub(r"<script>.*?</script>", " ", text, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
        return set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", text))

    md = literals(md_p.read_text(encoding="utf-8"))
    html = literals(html_p.read_text(encoding="utf-8"), strip_markup=True)
    for missing, where in ((md - html, "deck.html"), (html - md, "deck.md")):
        if missing:
            problems.append(f"deck parity: {sorted(missing)} present in the other rendering "
                            f"but absent from {where}")
    if not (md - html) and not (html - md):
        notes.append(f"deck.md and deck.html agree on all {len(md)} numeric literals")


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
    check_arithmetic(problems, notes)
    check_deck_parity(problems, notes)
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
