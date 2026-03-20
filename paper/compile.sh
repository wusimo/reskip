#!/bin/bash
# Compile the paper
export PATH="$HOME/.local/texlive/2026/bin/x86_64-linux:$PATH"

cd "$(dirname "$0")"

# First pass
pdflatex -interaction=nonstopmode main.tex

# Bibliography
bibtex main

# Second + third pass for references
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo ""
if [ -f main.pdf ]; then
    echo "SUCCESS: main.pdf generated ($(du -h main.pdf | cut -f1))"
else
    echo "FAILED: main.pdf not generated"
fi
