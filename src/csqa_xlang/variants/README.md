# `variants/` — build condition variants

**Responsibility:** combine the English `CSQAItem`s with their translations into the
evaluation conditions. Each variant is a list of `CSQAItem`s where the question
and/or choices are in a particular language, with **labels and gold key unchanged**.

**Conditions (see CLAUDE.md / `methods.tex`):**

| Variant | Question | Answers |
|---------|----------|---------|
| `en-en` | EN | EN (baseline) |
| `en-x`  | EN | X |
| `x-x`   | X  | X |
| `x-en`  | X  | EN |
| `mixed` | EN | choices split across two languages X₁/X₂ (extension) |
| `noun`  | EN with select nouns → X | EN (extension) |

**Design intent:**

- A variant is a pure transform over loaded items + translations — deterministic,
  no model calls.
- Fingerprint each built variant (see `csqa_xlang.manifest.content_hash`) and persist
  to `data/variants/` so eval and analysis reference an exact, hashable input.
- For `mixed`, fix the choice→language assignment with the global seed so it's
  reproducible; record the assignment.
