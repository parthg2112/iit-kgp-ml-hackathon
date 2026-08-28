# Prompt — recolour the deck

> Paste everything below the line into the project agent.
> Amends `support/pitch-deck-design-prompt.md`; where the two disagree, **this file wins**.

---

## Scope

One change to `docs/pitch/final/slides.html`: the colour system.

**Do not touch** the layout, grid, type scale, spacing, archetypes, content, speaker notes,
or the fonts. Typography and positioning are already right. This is a colour pass and
nothing else. `slides.md` is out of scope.

Fonts are already resolved — see the note at the end. Do not redo that work.

---

## The problem

The deck's `:root` declares eleven colour tokens, but `--blue`, `--green`, `--purple`,
`--gold`, `--green-light` and `--lilac` are **never referenced anywhere in the file**.
What actually renders is a cream ground, ink type, and coral-orange accents — which is
very close to Claude's own brand palette, and reads as machine-generated rather than
designed.

The cause is an instruction in the earlier spec that reserved those six colours "for data
series only". That instruction was wrong about the reference site. Measured from the
site's extracted data:

- **23 of its 45 slides carry a full-bleed colour field** — blue, green, purple, gold,
  orange-light, green-light, lilac, ink and cream-shade all appear as large fields.
- Each major section has a colour identity: Section 2.1 gold, Section 4.1 purple,
  Section 6.1 green-light, Section 7.1 lilac, Section 5.2 orange-light.

## The governing rule

Also measured from the same source — across all 45 slides the text colour is
**ink 466×, cream 64×, orange 6×, orange-light 2×**. So:

> **Colour lives in fields. Type is only ever `--ink` or `--cream`, whichever contrasts.**

The deck currently does the opposite: it leaves every field cream and colours the italic
display word orange on each title. That inversion is the whole problem.

## Section colours

Each slide takes one colour, applied to its title slab. **Slides 2 and 4 have no slab
today and must gain one**, built exactly like the existing three.

| # | Slide | Field | Type on it | Contrast |
| --- | --- | --- | --- | --- |
| 1 | The problem | `--ink` `#2e2e30` | `--cream` | 12.42 · AAA |
| 2 | Chemical understanding | `--blue` `#455ca0` | `--cream` | 5.85 · AA |
| 3 | ML methodology | `--purple` `#a376b2` | `--ink` | 3.74 · AA-large |
| 4 | Results | `--gold` `#c9a465` | `--ink` | 5.80 · AA |
| 5 | Failure analysis | `--green` `#5c7d3a` | `--cream` | 4.33 · AA-large |

Ratios are computed WCAG relative luminance over the exact token hexes, not estimates.
Slab type is 68 px display, so AA-large (≥3.0) is the applicable threshold and all five
pass. **The two marked AA-large must not be reused at body size** — they fail there.

## Implementation

Declare the pairing once per slide and let it cascade. Each slide's root is
`.frame > section.stage`; put the variables on `.stage`.

```css
/* section colour — one pairing per slide, cascaded to slab and rail */
.stage { --sec: var(--ink); --sec-on: var(--cream); }

.frame:nth-of-type(2) .stage { --sec: var(--blue);   --sec-on: var(--cream); }
.frame:nth-of-type(3) .stage { --sec: var(--purple); --sec-on: var(--ink);   }
.frame:nth-of-type(4) .stage { --sec: var(--gold);   --sec-on: var(--ink);   }
.frame:nth-of-type(5) .stage { --sec: var(--green);  --sec-on: var(--cream); }
```

Prefer an explicit `data-sec="1".."5"` attribute on each `.stage` over `nth-of-type` if
the frames are not direct siblings — the selector must not silently mis-target.

Then four edits:

**1. The slab takes the section colour.** Currently hard-coded ink at `.slab`:

```css
.slab {
  background: var(--sec);      /* was: var(--ink)   */
  color: var(--sec-on);        /* was: var(--cream) */
}
```

**2. Slides 2 and 4 gain a slab.** Their titles are currently bare `.t-title` on cream
(the `.it` spans at the two title sites without a `.slab` wrapper). Wrap each in
`<div class="slab">` and add `below-slab` to the following column, matching slides 1, 3
and 5 exactly. Verify the title still fits the 640 px slab width at 68/72 — if a line
breaks badly the slab may extend to full bleed (1280) as the spec allows, but do not
change the type size.

**3. `.hl` loses its colour entirely.** Delete both rules:

```css
.hl       { color: var(--orange); }        /* DELETE */
.slab .hl { color: var(--orange-light); }  /* DELETE */
```

Keep or strip the `hl` class in the markup — either is fine — but the italic word must
now be distinguished **only by being italic**. That is already a full typeface change
(`.it` swaps to the serif family against a grotesque), so the distinction survives
comfortably. This matches the source, where display type is never coloured.

Note this also covers `class="metric hl"` on slide 4 — that figure becomes ink. It does
not lose emphasis; it is already the largest thing on the slide, and slide 4 now carries
a gold slab.

This edit fixes a real defect. `--orange` on `--cream` is **2.95:1** and `--orange-light`
on `--cream` is **2.10:1** — both fail even the AA-large threshold. Any remaining inline
emphasis becomes ink at weight 400 against 300 body, the same device already used for
`<em>`.

**4. The rail's active dot takes the section colour.**

```css
.rail > span.on > span::after { background: var(--sec); }   /* was: var(--orange) */
```

## What stays orange, and why

**The slide-2 chart keeps series 1 `--ink` and series 2 `--orange`.** Do not change it.
The chart reads its colours through `getComputedStyle` (the `css()` helper feeding the
`accent` variable), so this needs no code change — just leave `--orange` bound to it.

This is deliberate. Ink and orange are the only pair in this palette separated in **both**
hue and lightness; blue or green as a second series would collide with the ink series at
2 px line weight. Orange inside a chart is not the Claude signal — orange as the deck's
identity was, and that is what is being removed.

`button:hover` in the screen-only controls bar may also keep `--orange`. It is not part
of any slide and never prints.

## Reserve set

Leave declared but unused, documented as available for future slides:
`--green-light` `#c3d3ad`, `--lilac` `#dbd8e1`, `--cream-shade` `#e8dfd4` (already in use
for table zebra), plus two blues the original token list missed and which you may add:
`#647cc3` and `#81a0cd`.

---

## Fonts — already resolved, do not redo

Commit `10aa819` replaced the previously vendored commercial webfonts with OFL faces, and
this has been verified:

| File | Family | Licence |
| --- | --- | --- |
| `docs/fonts/grotesque-variable.woff2` | **Hanken Grotesk 3.013** — confirmed from the font's own name table | OFL 1.1 |
| `docs/fonts/serif-italic-variable.woff2` | **Playfair Display Italic** — per `docs/fonts/README.md`; the file's name table is subset-stripped so it could not be read back directly | OFL 1.1 |

No commercial or trial font file remains anywhere in the repo. For the record, the faces
that were removed have since been identified from their name tables as **PolySans**
(Gradient) and **Tiempos Text** (Klim Type Foundry) — and the Tiempos files were *trial
builds* (`Test Tiempos Text`, vendor `test-fonts`), licensed for evaluation only and never
publishable. Removing them was necessary, not just cautious. Do not reintroduce them.

One piece of stale documentation to fix while you are here:
`support/pitch-deck-design-prompt.md` still instructs a reader to copy
`63a051e803d98e0027228e67_*.woff2` into `atw-grotesque-*.woff2`. Replace that section with
a pointer to `docs/fonts/README.md`, so nobody re-vendors the trial fonts by following it.

---

## Verification

- [ ] All five slides have a slab, each a different colour, matching the table above.
- [ ] **No text anywhere is any colour other than `--ink` or `--cream`.** Grep the file:
      `color:` should resolve only to those two, plus the chart's canvas calls and the
      screen-only `button:hover`.
- [ ] The rail dot on each slide matches that slide's section colour.
- [ ] The slide-2 chart still draws ink + orange.
- [ ] Every block on every slide bottoms out at **≤ y=672** (the bottom safe margin) —
      slides 2 and 4 are the ones at risk, since they gain a slab.
- [ ] `--print-to-pdf` produces **5 pages**, no clipped text, no missing glyphs — check
      `E₂/E₁`, subscripts, `τ`, `σ`, `ρ`, `→`, `≈`, and the en-dashes.
- [ ] Screenshots at 1280×720 and 1920×1080 look correct.
- [ ] Content unchanged: numeric-token multiset diff of rendered text, before vs after.
      **Zero numbers may change.**

Report anything you could not satisfy, and why, rather than silently deviating.
