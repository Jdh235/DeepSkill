"""RQ1 figure of the final human relevance labels (after re-annotation), per dataset.

    fig_relevant_per_task.pdf           relevant skills (graded >= 2) per task, split
                                        into reference and non-reference skills and
                                        by grade (3, 2)

SRA-Bench datasets first, then SRA-Bench (all) and Skill-Usage, which are never
pooled. Writes to figures/output/.

    python3 figures/scripts/generate_label_figures.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures" / "output"  # everything the paper includes
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from generate_rq1_figures import DISPLAY, load_reference  # noqa: E402
from human_labels import load_labels  # noqa: E402  (generate_rq1_figures puts labels/human/ on the path)

SRA = ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"]
GROUPS = [*((DISPLAY[ds], [ds]) for ds in SRA), ("SRA-Bench (all)", SRA), ("Skill-Usage", ["skill_usage"])]
INK, INK_2, MUTED, SURFACE = "#000000", "#000000", "#000000", "#ffffff"  # text and axis lines pure black, as in the paper


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / name}.pdf")


def main() -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update({
        # Computer Modern to match the LLNCS paper (bundled cmr10; text.usetex would need LaTeX installed)
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "text.usetex": False,
        "axes.formatter.use_mathtext": True,
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "savefig.facecolor": SURFACE,
    })
    labels = load_labels()
    reference = load_reference()  # (dataset, instance_id, skill_id) of every reference skill
    relevant_per_task(labels, reference)


def relevant_per_task(labels: list[dict], reference: set) -> None:
    """Mean number of relevant skills per task, split into reference and non-reference skills and by final grade (3, 2)."""
    # hue = reference (blue) vs non-reference (orange), shade = grade (dark 3, light 2). Checked with the dataviz
    # validator on the white page, in bar order: CVD and normal-vision separation, chroma and contrast all pass
    # (the earlier grade-3 orange #b8461d sat too close to the grade-2 orange, normal-vision dE 12.8 < 15)
    palette = {(True, 3): "#184f95", (True, 2): "#5a97e0", (False, 3): "#a33d16", (False, 2): "#eb6834"}
    names = {True: "Reference", False: "Non-reference"}
    pos = [0, 1, 2, 3, 4, 5.2, 6.4]
    fig, ax = plt.subplots(figsize=(4.8, 1.3))  # short to save page space (was 2.2 in, then 1.7 in)
    xmax = 0.0
    for (name, dsl), y in zip(GROUPS, pos):
        rows = [r for r in labels if r["dataset"] in dsl]
        tasks = len({r["instance_id"] for r in rows})
        n = {(ref, g): sum(((r["dataset"], r["instance_id"], r["skill_id"]) in reference) == ref and r["final"] == g
                           for r in rows) for ref in (True, False) for g in (2, 3)}
        n_ref, n_non = n[(True, 2)] + n[(True, 3)], n[(False, 2)] + n[(False, 3)]
        print(f"{name:16} relevant (graded >= 2): {n_ref} reference ({100 * n_ref / (n_ref + n_non):.0f}%), "
              f"{n_non} non-reference ({100 * n_non / (n_ref + n_non):.0f}%), {n_non / n_ref:.1f}x as many non-reference")
        left = 0.0
        for key in palette:  # reference 3, reference 2, non-reference 3, non-reference 2
            width = n[key] / tasks
            ax.barh(y, width, left=left, height=0.7, color=palette[key], edgecolor=SURFACE, linewidth=0.5,
                    label=f"{names[key[0]]}, grade {key[1]}" if y == pos[0] else None)
            left += width
        xmax = max(xmax, left)
    # the benchmark-level rows in italics, as in the tables. Through mathtext: matplotlib bundles no text-italic
    # Computer Modern, so fontstyle="italic" silently stays upright
    def italic(name: str) -> str:
        return re.sub(r"[^\s-]+", lambda m: rf"$\mathit{{{m.group()}}}$", name)
    ax.set_yticks(pos, [name if i < len(SRA) else italic(name) for i, (name, _) in enumerate(GROUPS)])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.05)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", length=2, colors=INK_2)
    ax.set_xlabel("Relevant skills per task", color=INK_2)
    # legend as a 2x2 key: rows reference / non-reference, columns grade 3 / grade 2
    key = ax.inset_axes([0.80, 0.30, 0.12, 0.30])
    for i, ref in enumerate((True, False)):
        for j, g in enumerate((3, 2)):
            key.add_patch(plt.Rectangle((j, 1 - i), 0.92, 0.88, color=palette[(ref, g)]))
    for j, g in enumerate((3, 2)):
        key.text(j + 0.46, 2.05, str(g), ha="center", va="bottom", fontsize=8, color=INK_2)
    # above the digits by a fixed 10 pt, so it doesn't overlap them however short the figure is
    key.annotate("Grade", xy=(0.96, 2.05), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                 fontsize=8, color=INK_2, annotation_clip=False)
    for i, ref in enumerate((True, False)):
        key.text(-0.15, 1.44 - i, names[ref], ha="right", va="center", fontsize=8, color=INK_2)
    key.set_xlim(0, 2)
    key.set_ylim(0, 2)
    key.axis("off")
    save(fig, "fig_relevant_per_task")


if __name__ == "__main__":
    main()
