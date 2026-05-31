# `analysis/` — metrics and analysis

**Responsibility:** turn cached `results/` outputs into the numbers and figures in the
paper. Reads raw outputs (never re-queries models).

**Metrics:**

- **Accuracy** per (model, condition, language).
- **Flip rate** between condition pairs (e.g. `en-en` → `en-x`): fraction of items
  whose selected label changes. This is the core diagnostic — also report flip
  *direction* (toward vs. away from gold).
- **Statistical significance** on deltas. Because conditions share the same items,
  use a **paired** test — McNemar for accuracy differences (`statsmodels`
  `mcnemar`). Report it whenever a delta is small (writing recs).

**Design intent:**

- Functions are pure over loaded result records → easy to test on tiny fixtures.
- Emit both a tidy table (CSV) and publication figures to `results/` for the paper's
  `tables/` and `figures/`.
- Keep self-explanatory captions in mind: the analysis should output exactly the
  numbers the paper tables need, labeled clearly.
