# Pitch materials

The submitted deck lives in `final/`.

| File | What it is |
|---|---|
| `slides.html` | **The deck.** 5 slides, 5 minutes. Open in Chrome, then Print with Layout: Landscape, Margins: None, Background graphics: on. |
| `slides.md` | The deck's markdown twin. `scripts/audit_pitch.py` reads this one, so keep its numbers in step with the HTML. |
| `final-pitch.pdf` | Exported deck, the file that is submitted. |
| `final-notebook.pdf` | Exported notebook. |

Fonts used by the deck are in `docs/fonts/` (both SIL OFL 1.1 licensed).

## Rebuilding the PDFs

```bash
scripts/nb2pdf.sh notebook/final.ipynb docs/pitch/final/final-notebook.pdf
```

## Consistency audit

Run `scripts/audit_pitch.py`. It must exit 0. It re-reads every quoted number live from
`artifacts/`, checks the submission hash, and refuses values that have been retracted.
