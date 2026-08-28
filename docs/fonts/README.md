# Fonts used by `docs/pitch/final/slides.html`

Two self-hosted variable webfonts, aliased in the deck as **ATW Grotesque** and **ATW Serif**.
The aliases are deliberate: the deck names roles, not families, so a face can be swapped
without touching the stylesheet.

| File here | Family | Licence |
|---|---|---|
| `grotesque-variable.woff2` | [Hanken Grotesk](https://fonts.google.com/specimen/Hanken+Grotesk) | SIL Open Font License 1.1 |
| `serif-italic-variable.woff2` | [Playfair Display](https://fonts.google.com/specimen/Playfair+Display), italic | SIL Open Font License 1.1 |

Both are latin-subset variable fonts fetched from Google Fonts, so a single file spans the
whole weight axis. The deck declares `font-weight: 100 900` on each `@font-face` and lets the
browser interpolate the 300 / 400 / 500 the design uses. Declaring a single fixed weight
against a variable file snaps every weight to one cut, which is why the range form matters.

## Licence

SIL OFL 1.1 permits redistribution, embedding and modification, including inside a submitted
PDF. Nothing here needs clearing.

An earlier revision of this deck self-hosted two commercial webfonts taken from a third
party's asset payload. They have since been identified from their name tables as **PolySans**
(Gradient) and **Tiempos Text** (Klim Type Foundry), and the Tiempos files were *trial builds*
(`Test Tiempos Text`, vendor `test-fonts`) licensed for evaluation only and never publishable.
Removing them was necessary, not merely cautious. Do not reintroduce them, and do not follow
the font section of any older copy of `support/pitch-deck-design-prompt.md`.

## Replacing a face

Swap the file, keep the filename, and re-render. The deck is absolutely positioned on a fixed
1280x720 canvas, so a face with different metrics can push content past the 672 bottom safe
margin or re-wrap the hero titles that size the ink slabs on slides 1, 3 and 5. Print all five
slides and check for clipping before trusting the change.
