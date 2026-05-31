# `data/` — dataset loading

Loads CommonsenseQA into a normalized `CSQAItem` (id, English question, label→choice
mapping, gold label).

**Two conventions enforced here (don't change silently):**

- **Default split is `validation`** (~1,221 items). `test` has withheld answer keys
  (leaderboard) → `load_csqa("test")` raises. All scored runs use `validation`.
- **Choices keyed by letter (A–E).** Translation and variant steps keep the same
  labels and ordering, only swapping choice *text*. The model later selects a
  *letter*, so scoring stays language-invariant.

Source: `tau/commonsense_qa` on the Hugging Face Hub.

```python
from csqa_xlang.data import load_csqa
items = load_csqa("validation", limit=20)   # list[CSQAItem]
```
