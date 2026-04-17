# Paper drafts

This directory contains **two** draft versions of the paper, corresponding to two different strategic framings. Both are maintained in parallel; the retrofit draft (`main.tex`) is the current main line going forward.

| File | Title | Core framing | Status |
|---|---|---|---|
| `main.tex` / `main.pdf` | "Retrofitting Pretrained Transformers for Zero-Cost Adaptive Computation via Attention Residuals" | Retrofit method is the main contribution; ReSkip is validation; ReLoop → future work; VLA via retrofitted VLM | **v2, main line** (matches `RETROFIT_PAPER_PLAN.md`) |
| `main_v1.tex` / `main_v1.pdf` | "Attention Residuals as Adaptive Computation Routers: Input-Dependent Depth for LLMs and Vision-Language-Action Models" | Three parallel contributions: ReSkip, ReLoop, VLA modality-aware skip | v1, kept for reference |

## Build

```bash
bash compile.sh            # builds main.pdf (v2 retrofit)
pdflatex main_v1.tex && bibtex main_v1 && pdflatex main_v1.tex && pdflatex main_v1.tex  # builds main_v1.pdf (v1)
```

## When to update which

- **`main.tex`**: all going-forward experiments (retrofit on Llama-2-7B, VLA on Qwen2-VL) fill into this. TODO markers indicate where data needs to land.
- **`main_v1.tex`**: only update if we need to revive the parallel-contributions framing (e.g., if retrofit fails and we fall back to from-scratch-only story). Currently frozen.

## Figures

All figures under `figures/` are shared between versions. See `figures/generate_reskip_figures.py` for the reproducible generation script (uses real 340M FineWeb-Edu results from `../outputs/`).
