"""A 'gallery' visualization of Hebrew answer-choice translations: clean vs broken.

Left column  — high-confidence CORRECT: choices where all three MT systems
               (Google / NLLB-3.3B / Opus) produce the same Hebrew string.
Right column — BROKEN: real failure modes found in the data — Opus Latin-script
               hallucinations, untranslated words, and within-question collisions
               (two distinct English choices mapping to one Hebrew string).

Hebrew is right-to-left; rendered via python-bidi so matplotlib lays it out
correctly. Writes report/figures/fig14_hebrew_translations.png.

Usage:  python -m scripts.hebrew_gallery
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    from bidi.algorithm import get_display
except ImportError:  # pragma: no cover
    def get_display(s):  # fallback: at least reverse pure-RTL words
        return s[::-1]

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures" / "fig14_hebrew_translations.png"

# Curated from the data (scripts.translation_agreement / build_consensus signals).
CLEAN = [  # all 3 MT systems agree -> high-confidence correct
    ("bank", "בנק"), ("mall", "קניון"), ("market", "שוק"),
    ("bookstore", "חנות ספרים"), ("train station", "תחנת רכבת"),
    ("pizza", "פיצה"), ("river", "נהר"), ("waterfall", "מפל"),
]
BROKEN = [  # (english, hebrew, system · failure mode)
    ("new york", "ניו יורקworld. kgm", "Opus · Latin hallucination"),
    ("north carolina", "צפון קרוליינהworld. kgm", "Opus · Latin hallucination"),
    ("hutch", "locia_ subjects. kgm", "Opus · garbled output"),
    ("cow carcus", "Carcus פרה", "Opus · word left in English"),
    ("get in line  +  stand in line", "לעמוד בתור", "Google · collision (2 → 1)"),
    ("bring cash  +  bring cash", "להביא מזומן", "duplicate choice → 1 string"),
]

GREEN, RED, INK, MUTED = "#2E8B57", "#C0392B", "#222222", "#888888"


def he(s: str) -> str:
    return get_display(s)


def card(ax, x, y, w, h, english, hebrew, color, tag=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.4, edgecolor=color, facecolor=color + "14"))
    ax.text(x + 0.02, y + h - 0.03, english, fontsize=11, color=INK,
            va="top", ha="left", style="italic")
    ax.text(x + w - 0.02, y + h * 0.40, he(hebrew), fontsize=20, color=color,
            va="center", ha="right", weight="bold")
    if tag:
        ax.text(x + 0.02, y + 0.022, tag, fontsize=8, color=RED, va="bottom", ha="left")


def main():
    n = max(len(CLEAN), len(BROKEN))
    fig, ax = plt.subplots(figsize=(12.5, 1.15 * n + 1.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    top = 0.90
    ch = 0.86 / n        # card height
    gap = ch * 0.18
    cw = 0.44
    ax.text(0.24, 0.955, "✓  clean  —  all 3 MT systems agree",
            fontsize=14, color=GREEN, ha="center", weight="bold")
    ax.text(0.76, 0.955, "✗  broken  —  hallucination / collision",
            fontsize=14, color=RED, ha="center", weight="bold")

    for i in range(n):
        y = top - (i + 1) * ch + gap / 2
        if i < len(CLEAN):
            en, hb = CLEAN[i]
            card(ax, 0.02, y, cw, ch - gap, en, hb, GREEN)
        if i < len(BROKEN):
            en, hb, tag = BROKEN[i]
            card(ax, 0.54, y, cw, ch - gap, en, hb, RED, tag)

    fig.suptitle("Hebrew answer-choice translations — clean vs broken",
                 fontsize=16, weight="bold", y=0.995)
    ax.text(0.5, 0.005,
            "English question is unchanged; only the choices are translated and the model picks a letter. "
            "Google (the main set) is largely clean; the worst breakage is Opus on Hebrew.",
            fontsize=8.5, color=MUTED, ha="center", va="bottom")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
