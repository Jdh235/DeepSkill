"""Bar charts of the 0-3 relevance grade distribution (raw counts) per rater.

Same data as table_judge_grades_dataset_<judge>: for each annotator (A1-A3,
first-round), the final human label and the LLM judge, the number of judged
task-skill pairs given each grade. Writes to figures/output/, for each judge in
generate_judge_tables.JUDGES:

    fig_judge_grades_<judge>.pdf          SRA-Bench and Skill-Usage

    python figures/scripts/generate_judge_grade_bars.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from generate_judge_tables import ANNOTATORS, BENCHMARKS, JUDGES, SHORT, load_pairs  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "output"  # everything the paper includes
OUT_DIR.mkdir(exist_ok=True)
GRADES = [0, 1, 2, 3]
RATERS = [*((SHORT[a], lambda p, i=i: p["humans"][i]) for i, a in enumerate(ANNOTATORS)),
          ("Final", lambda p: p["final"]), ("Judge", lambda p: p["judge"])]
# The comparison is final label vs judge, so those two take the lead colours (blue, orange); the three anonymous
# annotators are context, in one muted grey, told apart by their fixed order (A1-A3). Checked with the dataviz
# validator on the white page, all pairs: CVD and normal-vision separation and contrast pass (the grey's low
# chroma is deliberate). Shades of one hue for five raters cannot pass those checks.
GREY = "#898781"
COLOURS = {"A1": GREY, "A2": GREY, "A3": GREY, "Final": "#2a78d6", "Judge": "#eb6834"}
LEGEND = {"A1": "Annotators", "Final": "Final label", "Judge": "LLM judge"}  # one entry for the three greys
INK, INK_2, MUTED, GRID, SURFACE = "#000000", "#000000", "#000000", "#e6e5e1", "#ffffff"  # text and axis lines pure black, as in the paper


def panel(ax, title: str, P: list[dict]) -> None:
    width = 0.8 / len(RATERS)
    for k, (name, grade) in enumerate(RATERS):
        counts = [sum(grade(p) == g for p in P) for g in GRADES]
        xs = [g - 0.4 + width * (k + 0.5) for g in GRADES]
        # surface-coloured edge: a gap between adjacent bars, so the three same-colour annotator bars stay separate
        ax.bar(xs, counts, width=width, color=COLOURS[name], edgecolor=SURFACE, linewidth=0.8,
               label=LEGEND.get(name, "_nolegend_"), zorder=2)
    ax.set_title(f"{title} ($n$={len(P):,})", fontsize=9, color=INK, pad=4)
    ax.set_xticks(GRADES, [str(g) for g in GRADES])
    ax.set_xlabel("Grade", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.margins(y=0.05)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0, colors=INK_2)


def save(fig, name: str) -> None:
    handles, names = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.06), handlelength=1.2, columnspacing=1.4)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"Saved: {OUT_DIR / name}.pdf")


def main() -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update({
        # Computer Modern to match the LLNCS paper; text.usetex needs a LaTeX install, so use the bundled cmr10
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "text.usetex": False,
        "axes.formatter.use_mathtext": True,  # cmr10 has no minus glyph
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "savefig.facecolor": SURFACE,
    })
    for judge, judge_dir in JUDGES.items():
        pairs = load_pairs(judge_dir)
        suffix = f"_{judge}"

        fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.75))  # LLNCS text width; kept short to save page space
        for ax, (b, dsl) in zip(axes, BENCHMARKS.items()):
            panel(ax, b, [p for p in pairs if p["dataset"] in dsl])
        axes[0].set_ylabel("Task-skill pairs", color=INK_2)
        fig.tight_layout(w_pad=2)
        save(fig, f"fig_judge_grades{suffix}")
        plt.close(fig)


if __name__ == "__main__":
    main()
