"""Pick the answer per item by combining a model's predictions ACROSS translations.

For a given model we have its letter prediction under several independent choice
translations (Google / NLLB / Opus). This aggregates those per-item predictions
two ways, using whatever sources are present for that item ("if available"):

  * vote   — majority letter across the available translations; ties are broken by
             source priority (the order in --sources, Google first by default).
             A realistic ensemble you could actually deploy.
  * oracle — correct if ANY available translation yielded the gold letter. The
             ceiling: how much accuracy is recoverable purely by choosing the
             right translation per item (an upper bound, not deployable).

Reads each source's cached run:  results/<model>/en-x__<lang>[<suffix>]__*/outputs.jsonl
Writes:                          results/translation_ensemble.csv

Usage:
  python -m scripts.translation_ensemble --model-tag xlmr-ep6
  python -m scripts.translation_ensemble --model-tag mbert-ep6 --sources google,nllb,opus,consensus
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path

LANGS = ["ru", "es", "he"]
SUFFIX = {"google": "", "nllb": "-nllb", "opus": "-opus", "consensus": "-consensus"}


def load_run(results: Path, model: str, lang: str, source: str) -> dict[str, tuple[str, str]]:
    """id -> (pred, gold) for one (model, lang, source), or {} if absent."""
    pat = str(results / model / f"en-x__{lang}{SUFFIX[source]}__*" / "outputs.jsonl")
    hits = sorted(glob.glob(pat))
    if not hits:
        return {}
    out = {}
    with open(hits[0], encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = (r.get("pred"), r.get("gold"))
    return out


def combine(per_source: dict[str, dict], order: list[str]):
    """Per item, vote (priority tie-break) + oracle over available source preds."""
    ids = set().union(*(set(d) for d in per_source.values())) if per_source else set()
    n = vote_ok = oracle_ok = 0
    per_n = Counter()  # how many items had k sources available
    for cid in ids:
        avail = [(s, per_source[s][cid][0]) for s in order
                 if cid in per_source[s] and per_source[s][cid][0] is not None]
        if not avail:
            continue
        gold = next(per_source[s][cid][1] for s in order if cid in per_source[s])
        n += 1
        per_n[len(avail)] += 1
        # majority vote, ties broken by source priority (order of `avail`)
        cnt = Counter(p for _, p in avail)
        top = max(cnt.values())
        tied = {l for l, c in cnt.items() if c == top}
        vote = next(p for _, p in avail if p in tied)
        vote_ok += (vote == gold)
        oracle_ok += any(p == gold for _, p in avail)
    return n, vote_ok, oracle_ok, per_n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--sources", default="google,nllb,opus",
                    help="comma list, priority order (tie-break + 'gold' source first)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    results = Path(args.results_dir)
    order = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in order:
        if s not in SUFFIX:
            raise SystemExit(f"unknown source {s!r}; pick from {list(SUFFIX)}")

    rows = []
    pooled = {"n": 0, "vote": 0, "oracle": 0, "single": {s: [0, 0] for s in order}}
    print(f"model={args.model_tag}  sources={order}\n")
    for lang in LANGS:
        per_source = {}
        for s in order:
            d = load_run(results, args.model_tag, lang, s)
            if d:
                per_source[s] = d
        if not per_source:
            print(f"  [{lang}] no cached runs for {order} — skipped")
            continue
        # single-source accuracy (reference)
        single = {}
        for s, d in per_source.items():
            ok = sum(p == g for p, g in d.values())
            single[s] = ok / len(d)
            pooled["single"][s][0] += ok
            pooled["single"][s][1] += len(d)
        n, vote_ok, oracle_ok, per_n = combine(per_source, [s for s in order if s in per_source])
        pooled["n"] += n
        pooled["vote"] += vote_ok
        pooled["oracle"] += oracle_ok
        best_single = max(single.values())
        ref = " ".join(f"{s}={single[s]:.3f}" for s in order if s in single)
        print(f"  [{lang}] n={n}  {ref}  | vote={vote_ok/n:.3f}  oracle={oracle_ok/n:.3f}  "
              f"(oracle +{(oracle_ok/n - best_single)*100:.1f} pts over best single)")
        row = {"model": args.model_tag, "lang": lang, "n": n,
               "vote_acc": round(vote_ok / n, 4), "oracle_acc": round(oracle_ok / n, 4)}
        for s in order:
            row[f"acc_{s}"] = round(single[s], 4) if s in single else ""
        rows.append(row)

    if pooled["n"]:
        n = pooled["n"]
        singles = {s: pooled["single"][s][0] / pooled["single"][s][1]
                   for s in order if pooled["single"][s][1]}
        best = max(singles.values())
        print(f"\n  [pooled] n={n}  vote={pooled['vote']/n:.3f}  oracle={pooled['oracle']/n:.3f}  "
              f"(best single {best:.3f}; oracle ceiling +{(pooled['oracle']/n - best)*100:.1f} pts)")
        row = {"model": args.model_tag, "lang": "pooled", "n": n,
               "vote_acc": round(pooled["vote"] / n, 4),
               "oracle_acc": round(pooled["oracle"] / n, 4)}
        for s in order:
            row[f"acc_{s}"] = round(singles[s], 4) if s in singles else ""
        rows.append(row)

    if rows:
        out = Path(args.results_dir) / "translation_ensemble.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
