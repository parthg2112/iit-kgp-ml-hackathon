#!/usr/bin/env bash
# Render a markdown doc to a shareable PDF.
#
#   scripts/md2pdf.sh docs/pitch/support/STUDY_GUIDE.md
#   scripts/md2pdf.sh docs/pitch/support/STUDY_GUIDE.md ~/Desktop/study-guide.pdf
#
# Uses pandoc + XeLaTeX. XeLaTeX rather than pdflatex because these documents contain
# Unicode that pdflatex cannot typeset: sigma, tau, subscripts, times, >=, arrows.
set -euo pipefail

SRC="${1:?usage: md2pdf.sh <input.md> [output.pdf]}"
OUT="${2:-${SRC%.md}.pdf}"

# The source headings already carry their own numbers ("## 1.", "### 1.1"), so LaTeX
# numbering is left off to avoid "1.1  1. The problem". --shift-heading-level-by=-1 promotes
# the h2 sections to chapters-in-spirit so the title is not repeated as a section.
pandoc "$SRC" -o "$OUT" \
  --pdf-engine=xelatex \
  --toc --toc-depth=2 \
  --shift-heading-level-by=-1 \
  -V documentclass=article \
  -V papersize=a4 \
  -V geometry:margin=2.2cm \
  -V fontsize=11pt \
  -V linkcolor=RoyalBlue \
  -V urlcolor=RoyalBlue \
  -V colorlinks=true \
  -V mainfont="Georgia" \
  -V monofont="Consolas" \
  -V linestretch=1.15 \
  -V "header-includes=\\raggedbottom" \
  --highlight-style=tango \
  --metadata title="$(head -1 "$SRC" | sed 's/^#\+ *//')" \
  --metadata author="Team Claude ke Chhatore" \
  --metadata date="$(date +'%d %B %Y')"

echo "wrote $OUT"
