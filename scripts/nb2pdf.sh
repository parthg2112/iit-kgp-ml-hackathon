#!/usr/bin/env bash
# Render the notebook to a shareable PDF.
#
#   scripts/nb2pdf.sh
#   scripts/nb2pdf.sh notebook/final.ipynb ~/Desktop/notebook.pdf
#
# pandoc reads .ipynb natively and typesets the 42 LaTeX spans through real TeX. The
# nbconvert -> HTML -> Chrome route was tried and rejected: its template needs MathJax
# from a CDN, which does not load under file://, so every equation came out blank --
# including the reaction scheme. Do not switch back without checking the math renders.
set -euo pipefail

SRC="${1:-notebook/final.ipynb}"
OUT="${2:-${SRC%.ipynb}.pdf}"

# Georgia has no U+2192 or U+221D; map them to their math equivalents rather than
# changing the body font, which is shared with md2pdf.sh.
GLYPHS='\usepackage{newunicodechar}\newunicodechar{→}{\ensuremath{\rightarrow}}\newunicodechar{∝}{\ensuremath{\propto}}\raggedbottom'

pandoc "$SRC" -o "$OUT" \
  --pdf-engine=xelatex \
  --toc --toc-depth=2 \
  -V documentclass=article \
  -V papersize=a4 \
  -V geometry:margin=2cm \
  -V fontsize=10pt \
  -V linkcolor=RoyalBlue \
  -V urlcolor=RoyalBlue \
  -V colorlinks=true \
  -V mainfont="Georgia" \
  -V monofont="Consolas" \
  -V linestretch=1.1 \
  -V "header-includes=$GLYPHS" \
  --highlight-style=tango \
  --metadata author="Team Claude ke Chhatore" \
  --metadata date="$(date +'%d %B %Y')" \
  2>&1 | grep -v "security risk" || true

[ -s "$OUT" ] || { echo "FAILED: no PDF at $OUT" >&2; exit 1; }
echo "wrote $OUT"
