# Brain-Motivation Validation Experiments — design doc

Purpose: give the paper's brain-inspired framing direct empirical support so
reviewers can't dismiss it as "just neuroscience packaging on a layer-skip
method."

Two experiments. Both reuse existing checkpoints and existing infrastructure;
both run on a single H100 in well under an hour. **Not run yet** — see commands
at the bottom when ready.

---

## Experiment B (primary): difficulty-dependent depth

**Hypothesis (B-H1).** Skip count is *negatively* correlated with token
prediction difficulty. Easy tokens (low NLL) are computed shallowly; hard
tokens (high NLL) recruit more block computation.

This is the direct transformer analogue of Kar et al.\ 2019: hard images need
more recurrent processing, easy images do not. If our retrofit obeys this
relationship, the brain-inspired framing is grounded in measured behaviour
rather than rhetorical packaging.

**Setup.**

- Checkpoint: 2B retrofit `H_r256_5k` (the headline) with the
  `recent_weight_gt` skip rule at the recommended operating point
  ($\mathcal{P}=\{4,6,11\}$, $q=0.85$, $M_{\max}=2$).
- Data: 1{,}000 LAMBADA prefixes (already used for the dynamic-skip eval; same
  distribution — no risk of overfitting calibration).
- For each prefix: run a single dynamic-skip forward; record
  `(per_token_nll, total_skip_count, hardest_token_nll)` per example.
- Aggregate: bucket prefixes into easy / medium / hard by `mean nll`
  (terciles), report `mean(skip_count)` per bucket. Also compute Spearman
  $\rho$ between (NLL, skip\_count) at the example level and the per-token level.

**Expected pattern (success criterion).**

- Easy bucket: avg skip $\approx 1.5$ (more skips, since router collapses
  $\alpha$ onto the predecessor for easy continuations)
- Hard bucket: avg skip $\approx 0.5$ (fewer skips, router spreads $\alpha$
  to refine the prediction)
- Spearman $\rho < -0.15$ at $p < 0.001$ (1k prefixes is enough)

**Fallback if no correlation.**

If $\rho \approx 0$ or positive: retreat to weakened framing in §1 — drop the
"transformer reproduces the property" sentence, keep brain-inspired only as
design philosophy. The retrofit story does not depend on this.

If $\rho$ is in the right direction but small ($-0.05$ to $-0.10$): keep the
result with explicit caveat ("a small but significant correlation"). Add a
discussion note that the existing static $\mathcal{P}=\{4,6,11\}$ restricts
where input-dependence can show up; a per-block-eligible variant would be a
cleaner test.

**Deliverables.**

- `outputs/standard/analysis/brain_skip_difficulty.csv` — per-prefix table
- `paper/figures/brain_skip_difficulty.pdf` — scatter + binned means
- 1 paragraph in §3 (after the 340M Pareto), 1 sentence in §1

---

## Experiment C (supporting): cross-modality routing distribution

**Hypothesis (C-H1).** Per-block $\alpha$ distribution differs across modalities
(text / VL / action). The retrofit's router has implicitly developed
modality-specific pathways — analogous to dorsal/ventral cortical
specialisation~\citep{buschman2007topdown,bastos2012canonical}.

We *already have indirect evidence*: a LAMBADA-calibrated skip schedule
applied to LIBERO drops Spatial accuracy from $99.5\%$ to $64\%$
(\S\texttt{vla\_future}). That tells us the router emits different $w_{\text{recent}}$
distributions on text vs.\ action. C-H1 just makes this direct and visual.

**Setup.**

- Checkpoint: H\_r256\_5k (same as B) and the VLA H\_4B\_r256\_5k (for action
  modality).
- Data:
  - Text: 200 LAMBADA prefixes
  - VL: 200 MMStar examples (image+question)
  - Action: 200 LIBERO\_Spatial frames (image+language instruction passed
    through the VLA forward; `install_recording_router` already exists in
    `analyze_vla_reskip_2b_l4.py`)
- For each modality, dump per-block $\alpha$ across all positions; aggregate
  to a per-block histogram of $w_{\text{recent}}(n)$.

**Expected pattern.**

- Text: $w_{\text{recent}}$ distributions per block tightly concentrated;
  high-$q$ tail above the LAMBADA-calibrated $\tau_n$.
- VL: shifted — distinguishable by KL but not catastrophically so
  (recall MMBench $+3.5$pp gain shows VL routing works).
- Action: heavy tail / wider spread on late blocks (consistent with
  motor-planning needing deeper compute, action tokens routing to late blocks).

**Quantitative summary.**

- Per-block KL$(p_\text{text} \,\|\, p_\text{VL})$, KL$(p_\text{text} \,\|\, p_\text{action})$
- One stacked-histogram figure with 3 colours per block.

**Fallback.** If KLs are tiny (action $\approx$ text), modality-specific routing
isn't really happening structurally; fall back to "modality-aware skip is
future work because of distribution shift" (which is what we already say).

**Deliverables.**

- `outputs/standard/analysis/alpha_by_modality.npz` — raw per-modality samples
- `paper/figures/alpha_by_modality.pdf` — per-block stacked histogram
- 1 paragraph in `app:alpha_modality` referenced from the related-work
  brain-inspired paragraph and the VLA modality-aware future work.

---

## Why these two are sufficient

- B grounds the *computational* brain claim ("hard inputs recruit deeper
  processing")
- C grounds the *anatomical* brain claim ("modality-specific pathways")
- Together they cover the two parts of the §1 motivation that cite
  Kar/Lamme + Buschman/Bastos respectively.

A reviewer who asks "what's brain-inspired about your method besides the
opening paragraph?" can be pointed to B-Fig and C-Fig.

---

## When ready: exact commands

```bash
# Experiment B
python retrofit/analysis/brain_skip_difficulty.py \
  --state-path retrofit/saves/H_r256_5k/state.pt \
  --num-blocks 14 \
  --calib-n 32 \
  --eval-n 1000 \
  --eligible "4,6,11" --quantile 0.85 --max-skips 2 \
  --out-csv outputs/standard/analysis/brain_skip_difficulty.csv \
  --out-fig paper/figures/brain_skip_difficulty.pdf

# Experiment C
python retrofit/analysis/alpha_by_modality.py \
  --vlm-state-path retrofit/saves/H_r256_5k/state.pt \
  --vla-state-path starVLA/saves/H_4B_r256_5k/checkpoint.pt \
  --lambada-n 200 --mmstar-n 200 --libero-n 200 \
  --out-npz outputs/standard/analysis/alpha_by_modality.npz \
  --out-fig paper/figures/alpha_by_modality.pdf
```

Both should finish in under 30 GPU-min on 1×H100. Run only after the next
paper revision lands.
