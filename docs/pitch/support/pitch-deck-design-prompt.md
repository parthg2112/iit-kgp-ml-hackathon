# Prompt — rework the pitch deck onto the "Advocacy Through Walls" design system

> Paste everything below the line into your project agent.
> **Attach both reference screenshots** (the title screen, and the "Introduction / The Experts"
> two-column screen). Optionally also attach `DESIGN_TOKENS.md` from the clone repo.

---

## Role

You are redesigning an existing HTML pitch deck for a formal hackathon submission. You are
**restyling and re-laying-out**, not rewriting. Every claim, number, heading and body sentence
already in the deck must survive; if a slide is over-full, tighten wording, never delete a point.

## What you are matching

The design language comes from a single reference site. Two screenshots are attached:

- **Screenshot 1 — the title screen.** This is the primary reference. Study: the enormous
  three-line title set flush-left in a geometric grotesque, with **exactly one line in a serif
  italic**; the vast empty left field; the cream ground; the single hard-edged orange circle; the
  small caps-height credit line pinned bottom-left. Note how little is on the slide and how
  confident it reads because of it.
- **Screenshot 2 — a content screen.** Use it wherever a slide has parallel items. Study: the
  persistent slim header bar (disc marker, section label, roman-numeral rail with an active dot);
  the two-column list where each row is a label paired with a short description; the thin vertical
  divider between columns with a small orange dot on it; body copy at a genuinely small size with
  generous line-height; links in orange.

Take from these: **the type scale, the palette, the whitespace discipline, the flush-left editorial
rhythm, and the header chrome.** Do not take: the illustrations, the brick artwork, the paper
texture, or the hand-lettered name images (those are PNGs in the original, not a font — replace
them with type).

## Hard constraints

1. **Formal.** This is a hackathon submission judged on substance. No illustration, no stock
   photography, no gradients, no drop shadows, no glassmorphism, no rounded-everything.
2. **The data is the hero.** Charts and diagrams will be inserted later and must be the loudest
   thing on any slide they appear on. Every other element stays quiet.
3. **Restraint over decoration.** Only three decorative motifs are permitted, specified below.
   Nothing else.
4. **One accent per slide.** At most one orange element on any given slide. One.
5. **One italic per slide, maximum.** The serif italic is for a single emphasized word in a title.
   Never for body copy, never for two words on the same slide.
6. **16:9**, authored on a fixed **1280 × 720** canvas and uniformly scaled to the viewport.

## Design tokens — use verbatim

```css
:root {
  /* ground + ink */
  --ink:          #2e2e30;   /* all text; full-bleed divider background */
  --cream:        #fbf4ec;   /* default slide background */
  --cream-shade:  #e8dfd4;   /* table zebra, subtle fills, disabled */

  /* accent — see usage rules */
  --orange:       #ef6423;   /* THE accent: links, active dot, one highlight per slide */
  --orange-light: #f39454;

  /* reserved for data series only */
  --blue:         #455ca0;
  --green:        #5c7d3a;
  --purple:       #a376b2;
  --gold:         #c9a465;
  --green-light:  #c3d3ad;
  --lilac:        #dbd8e1;

  /* type */
  --font-display: "ATW Grotesque", ui-sans-serif, system-ui, sans-serif;
  --font-serif:   "ATW Serif", ui-serif, Georgia, serif;

  /* canvas + grid */
  --canvas-w: 1280px;
  --canvas-h: 720px;
  --margin:   48px;
  --col-3:    288px;  --gutter-3: 160px;
  --col-2:    560px;  --gutter-2:  64px;
  --baseline:   8px;

  /* radii — the source is nearly square-cornered; keep it that way */
  --r-hairline: 2px;
  --r-card:    24px;
  --r-pill:    40px;
}
```

### Colour rules

- `--cream` is the background of every content slide. `--ink` is the background of section
  dividers only.
- `--ink` is the only text colour on cream. `--cream` is the only text colour on ink.
- `--orange` appears at most **once per slide**: a link, the active rail dot, a single highlighted
  metric, or the divider dot. Never as a fill behind text.
- **The blue / green / purple / gold set is reserved for chart data series.** Do not use them for
  typography, rules, or backgrounds. Recommended series order:
  `--ink → --orange → --blue → --green → --purple → --gold`, with
  `--orange-light / --green-light / --lilac / --cream-shade` as their light fills.

## Fonts

Use the two faces from the reference site. Alias them so the deck is not coupled to the source's
internal names, and confirm the licence permits redistribution before you submit the file — they
are licensed webfonts, not open-source.

```css
/* geometric grotesque — all UI, headings, body */
@font-face { font-family:"ATW Grotesque"; font-weight:300; font-style:normal;
             src:url("../fonts/atw-grotesque-300.woff2") format("woff2"); font-display:swap; }
@font-face { font-family:"ATW Grotesque"; font-weight:400; font-style:normal;
             src:url("../fonts/atw-grotesque-400.woff2") format("woff2"); font-display:swap; }
@font-face { font-family:"ATW Grotesque"; font-weight:500; font-style:normal;
             src:url("../fonts/atw-grotesque-500.woff2") format("woff2"); font-display:swap; }

/* serif — italic display accent only */
@font-face { font-family:"ATW Serif"; font-weight:400; font-style:italic;
             src:url("../fonts/atw-serif-400-italic.woff2") format("woff2"); font-display:swap; }
@font-face { font-family:"ATW Serif"; font-weight:500; font-style:italic;
             src:url("../fonts/atw-serif-500-italic.woff2") format("woff2"); font-display:swap; }
```

Source files (rename on copy):

| Copy from | Rename to |
| --- | --- |
| `63a051e803d98e0027228e67_n3.woff2` | `atw-grotesque-300.woff2` |
| `63a051e803d98e0027228e67_n4.woff2` | `atw-grotesque-400.woff2` |
| `63a051e803d98e0027228e67_n5.woff2` | `atw-grotesque-500.woff2` |
| `63a2e064e55aa80038f09228_i4.woff2` | `atw-serif-400-italic.woff2` |
| `63a2e064e55aa80038f09228_i5.woff2` | `atw-serif-500-italic.woff2` |

Weight discipline, taken from the source: **300 for body, 500 for headings, 400 for UI labels.**
There is no bold. Do not introduce 600 or 700.

## Type scale

Absolute px on the 1280 canvas. Line-heights are absolute, never ratios.

| Token | Size / line-height | Weight | Use |
| --- | --- | --- | --- |
| `display-xl` | 224 / 224 | 500 | Deck title; the numeral on a section divider |
| `display-l` | 200 / 200 | 500 | Single-word statement slides |
| `hero` | 132 / 138 | 500 | Title-screen lines (screenshot 1) |
| `title` | 68 / 72 | 500 | Slide title; section divider title |
| `subhead` | 38 / 46 | 500 | Secondary line under a slide title |
| `heading` | 20 / 22 | 500 | Column and card headings |
| `body` | 16 / 21 | 300 | All body copy |
| `label` | 16 / 20 | 400 | Header bar label, rail numerals, chart axis labels |
| `micro` | 14 / 18 | 400 | Captions, footnotes, sources, credit line |

Rules:

- Body is **300 weight at 16px**. It will feel small. That is correct — it is what buys the
  whitespace. Do not inflate it.
- Titles are flush-left, never centred. Body is flush-left, ragged right, never justified.
- Letter-spacing is `0` everywhere. The faces are already drawn for this.
- Never set body copy wider than `--col-2` (560px). Two columns beat one wide one.

## Vertical rhythm

Everything snaps to an **8px baseline**.

- Slide title block starts at `y = 96`.
- Rule → heading: **10px**.
- Heading → its body: **21px**.
- Row pitch in a multi-row grid: **236px**.
- Bottom safe margin: **48px**. Nothing but the footer enters it.

## The three permitted motifs

**1 — Hairline rule above a heading.** A `1px` `--ink` rule spanning the full column width, sitting
10px above the heading's cap. This is the workhorse; it is what makes the deck read as editorial
rather than as slides. Use it on every column heading and card heading.

```
────────────────────────────
Person-Centered Advocacy          ← heading, 20/22, weight 500
                                  ← 21px
inside and outside advocates      ← body, 16/21, weight 300
working together toward the
abolition of unjust policies…
```

**2 — Header bar + roman-numeral rail.** A persistent 70px cream bar on every content slide, with a
`1px --ink` hairline along its bottom edge.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ●   Introduction                    I.  II.  III.  IV.  V.  VI. VII.│
│                                          •                           │
└──────────────────────────────────────────────────────────────────────┘
```

- 48px `--ink` disc at `(48, 12)` — a static marker, not a menu.
- Section label at `x = 112`, `label` token.
- Rail right-aligned to the 48px margin, first item at `y = 24`, items 46px apart, `label` token.
- Active section: an **8px `--orange` dot** centred 14px below its numeral. Inactive numerals stay
  full-strength `--ink` — do not dim them.
- The bar is **omitted** on the title slide and on all section dividers.

**3 — The orange dot.** A single 30px `--orange` circle, used as either the divider marker between
two columns (screenshot 2) or a lone punctuation mark in empty space (screenshot 1). Never more
than one per slide, never behind text.

Everything else — icons, borders, boxes, badges, arrows, shadows — is prohibited.

## Slide archetypes

Map every existing slide onto one of these. Do not invent a ninth.

**A · Title.** Cream. No header bar. Title set on 2–3 lines at `hero`, flush-left at
`x = 48`, first baseline near `y = 168`. **Exactly one line in `--font-serif` italic** — pick the
line that carries the idea, as "Through" does in screenshot 1. Team name / event / date bottom-left
at `micro`. One orange dot in the right two-fifths, well away from the type. At least half the
slide stays empty.

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Realtime Fraud                                                     │
│   Detection at                          ← one line in serif italic   │
│   the Edge                                                           │
│                                                        ●             │
│                                                                      │
│   Team Northstar · HackXYZ · Mar 2026                                │
└──────────────────────────────────────────────────────────────────────┘
```

**B · Section divider.** Full-bleed `--ink`. No header bar. Roman numeral at `display-xl` in
`--cream`, section title at `title` beneath it, both flush-left at `x = 48`. Nothing else.

**C · Statement.** Cream, header bar. One sentence at `title`, maximum twelve words, flush-left,
occupying the upper-left two-thirds. The rest empty. Use for the problem statement and the ask.

**D · Three-column.** Cream, header bar. Three `--col-3` columns at `x = 48 / 496 / 944`. Each:
hairline rule, `heading`, body. Up to two rows at a 236px pitch. This is the workhorse for
features, principles, and comparisons.

**E · Two-column list.** Cream, header bar. Directly from screenshot 2. Two `--col-2` columns at
`x = 48 / 672`. Each row is a bolded label (`heading`) with its description (`body`) beside or
beneath it. Optional `1px --ink` vertical divider at the midline with the orange dot on it. Use for
team, roles, tech stack, roadmap.

**F · Chart.** Cream, header bar. Slide title at `title` top-left. **The chart occupies at least
60% of the slide area.** One takeaway sentence at `body` directly beneath the chart, never beside
it. Source note at `micro`, bottom-left. Chart uses the reserved series palette, `--ink` axes at
`1px`, `label` axis type, **no gridlines** — or if unavoidable, `--cream-shade` at `1px`.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ●   IV. Results                          I. II. III. IV. V. VI. VII.│
├──────────────────────────────────────────────────────────────────────┤
│  Latency under load                                                  │
│                                                                      │
│    ┌────────────────────────────────────────────────────┐            │
│    │                                                    │            │
│    │              [ chart — ≥60% of slide ]             │            │
│    │                                                    │            │
│    └────────────────────────────────────────────────────┘            │
│    p99 stays under 40 ms to 10× projected peak traffic.              │
│    Source: internal load test, 2026-03-04                            │
└──────────────────────────────────────────────────────────────────────┘
```

**G · Metric.** Cream, header bar. One to three figures at `display-l`, each with a `body` caption
beneath. **At most one figure in `--orange`**; the rest `--ink`.

**H · Closing.** Same construction as A, different words. Contact details at `micro`.

## Prohibitions

- No centred text anywhere except the numeral on a divider.
- No bullet characters. Structure comes from columns and hairline rules. If a list is unavoidable,
  use a hanging `--ink` en-dash.
- No bold. 500 is the heaviest weight in the deck.
- No colour on text other than `--ink`, `--cream`, and a single `--orange` emphasis.
- No box outlines or card borders. Separation comes from whitespace and hairlines.
- No slide with more than **one** idea on it. Split before you shrink.
- No transitions or scroll animation. This deck will be presented and may be exported to PDF.
- Do not fill empty space. The empty left field in screenshot 1 is the design.

## Acceptance checklist

Before you report done, verify every item:

- [ ] Every original content point is still present.
- [ ] Every slide maps to exactly one archetype (A–H).
- [ ] Body copy is 16/21 at weight 300 everywhere; no instance of weight 600/700.
- [ ] The serif italic appears at most once per slide and never in body copy.
- [ ] No slide has two orange elements.
- [ ] Every measurement is a multiple of 8, or is a documented token value.
- [ ] The header bar is present on all content slides and absent on title and dividers.
- [ ] Each chart slide gives the chart ≥60% of the area and has a one-sentence takeaway.
- [ ] Deck renders correctly at 1280×720 and scales cleanly to 1920×1080.
- [ ] Prints to PDF with no clipped text and no missing glyphs.
- [ ] Fonts are self-hosted, aliased, and licence-checked.
