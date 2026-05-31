# `paper/` — the report

The 6-page (11pt) ACL-style report. Structured so it works **both** in Overleaf
and when compiled locally.

## Structure

- `main.tex` — the document shell: title, authors, abstract, `\input`s the
  sections, and the bibliography.
- `sections/*.tex` — one file per section (`introduction`, `related_work`,
  `methods`, `experiments`, `conclusion`, `appendix`). These are **style-agnostic**
  (plain `\section{...}` content), so they drop into any template unchanged.
- `references.bib` — bibliography. Three known entries are pre-filled; **verify
  every entry against the [ACL Anthology](https://aclanthology.org/) and cite the
  conference/journal version, not arXiv** (per the course writing recommendations).
- `figures/`, `tables/` — assets.

## Working in Overleaf (recommended)

The course recommends the ACL template. Easiest path:

1. In Overleaf, start a new project from the **official ACL template** (Menu →
   New Project → Templates → search "ACL"), *or* download the style files from
   <https://github.com/acl-org/acl-style-files>.
2. Replace the template's body with our `main.tex` and upload the `sections/`
   folder and `references.bib`.
3. Set the document to the **camera-ready / final** option only when submitting;
   keep `review` off (this is a course report, not a blind submission) — see the
   comment at the top of `main.tex`.

## Compiling locally

`main.tex` expects the ACL style files (`acl.sty`, `acl_natbib.bst`) in this
directory. Vendor them from the repo above, then:

```bash
cd paper
latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main x2
```

If you don't want to vendor the ACL files yet, switch the documentclass line at
the top of `main.tex` to plain `article` (a comment there explains how) to draft
content before wiring up the template.

## Length & format reminders

- 6 pages, 11pt, **excluding** references and appendix.
- Include a figure/example in the Introduction (recommended) — e.g. one CSQA item
  shown across the `EN-EN` / `EN-X` / `X-X` / `X-EN` conditions.
- Self-explanatory table/figure captions.
- Report statistical significance on small deltas.
