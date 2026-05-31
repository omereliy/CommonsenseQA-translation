# `translation/` — translate answer choices

**Responsibility:** given English `CSQAItem`s, produce per-language translations of
the **answer choices** (and, for the extension, selected **question nouns**).
Questions otherwise stay English.

**Design intent (shape as a team):**

- **Pluggable backend.** Expose a small interface (e.g. `translate(texts, src, tgt) ->
  list[str]`) with interchangeable implementations. The backend choice — MT
  (e.g. NLLB / DeepL / Google) vs LLM-based — is a *pending decision* (CLAUDE.md).
- **Cache translations** to `data/translated/` keyed by (text, src, tgt, backend) so
  the slow/paid step runs once and is reproducible. Treat the cache as derived data.
- **Preserve labels & ordering.** Translate only the choice *text*; keep A–E labels.
- **Meaning preservation.** Support exporting a small subset for optional human
  validation that translated choices keep the original sense (especially for
  near-synonym distractors, where a sloppy translation could change the answer).

Open question to resolve: how to handle multi-word choices and choices that are
proper nouns / culturally specific. Document the policy here once decided.
