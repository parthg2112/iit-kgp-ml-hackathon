# Pitch materials

Three folders, by what each is for.

```
final/      what gets submitted and presented
support/    depth behind the deck: Q&A, appendix, study guide
```

Source material we did not write lives one level up, in `docs/reference/`: the problem
statement, the organisers' offline-round guidelines, and the original team brief.

## final/

| File | What it is |
|---|---|
| `slides.html` | **The deck.** 5 slides, 5 minutes. Open in Chrome, then Print with Layout: Landscape, Margins: None, Background graphics: on. |
| `slides.md` | The deck's markdown twin. `scripts/audit_pitch.py` reads this one, so keep its numbers in step with the HTML. |
| `final-pitch.pdf` | Exported deck, the file that is submitted. |
| `final-notebook.pdf` | Exported notebook. Generated, not tracked. |

## support/

| File | What it is |
|---|---|
| `deck.md`, `deck.html` | The 12-slide long version, kept as the answer bank behind the 5 slides. Audited as a pair: every numeric literal must match. |
| `appendix.md` | Technical appendix: parameters, protocols, identifiability, rejected alternatives. |
| `qa_bank.md` | Anticipated questions with measured answers. |
| `STUDY_GUIDE.md` | Takes the team from zero to fluent on the reactor chemistry and on our solution. |
| `pitch-deck-design-prompt.md` | The design brief the deck was built against. |

## Rebuilding the PDFs

```bash
scripts/md2pdf.sh docs/pitch/support/STUDY_GUIDE.md
scripts/nb2pdf.sh notebook/final.ipynb docs/pitch/final/final-notebook.pdf
```

## Before submitting

Run `scripts/audit_pitch.py`. It must exit 0. It re-reads every quoted number live from
`artifacts/`, checks the submission hash, and refuses values we have retracted.
