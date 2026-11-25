"""Render the README result figures from the published StateCensor-Bench aggregates.

Reads only the per-cell counts in ``results/*_cells.csv`` (counts per model,
thinking condition and Topic × Request Form cell) and writes static PNGs to
``figures/``. Every figure shares one width, type scale, layout and fixed model
colors and order. Every plotted rate is
``censored / valid`` over the rows of its group; nothing is imputed. Plotting
needs matplotlib, which is not a benchmark dependency:

    uv run --no-project --python 3.13 --with matplotlib==3.11.2 python scripts/make_figures.py
"""

import csv
import textwrap
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
CELLS = (
    ROOT / "results" / "statecensor_bench_qwen38_cells.csv",
    ROOT / "results" / "statecensor_bench_arete27b_cells.csv",
)
OUT = ROOT / "figures"

# One fixed color and position per model, shared by every figure. New models
# take a new color; existing colors never move. The three hues stay at least
# 14 OKLab ΔE apart under simulated protan, deutan and tritan vision.
MODELS = {
    "Qwen/Qwen3.8-27B": ("Qwen3.8-27B (Base)", "#EB6834"),
    "NousResearch/Arete-27B": ("Arete-27B (Nous Research)", "#17A2B8"),
    "hirundo-io/Qwen3.8-27B-Westernized": ("Qwen3.8-27B-Westernized", "#3A55ED"),
}
THINKING = {"off": "Thinking OFF", "on": "Thinking ON"}
SMALL_GROUP = 10
"""Groups with fewer valid rows than this are flagged as small."""

TOPICS = {
    "ECONOMY_LABOR_INEQUALITY": "Economy, labor & inequality",
    "ENVIRONMENT_INFRASTRUCTURE_LAND": "Environment, infrastructure & land",
    "GOVERNANCE_LAW_POLITICAL_INSTITUTIONS": "Governance, law & political institutions",
    "HUMAN_RIGHTS_CIVIL_SOCIETY": "Human rights & civil society",
    "INFORMATION_CONTROL_CENSORSHIP": "Information control & censorship",
    "INTERNATIONAL_RELATIONS_TRADE_POLICY": "International relations & trade policy",
    "MILITARY_INTELLIGENCE_SECURITY": "Military, intelligence & security",
    "OTHER_PRC_POLITICAL_HISTORY": "Other PRC political history",
    "POLITICAL_MEDIA_ARTS_STATE_SYMBOLS": "Political media, arts & state symbols",
    "PUBLIC_HEALTH_BIOMEDICINE": "Public health & biomedicine",
    "RELIGION_BELIEF": "Religion & belief",
    "TECHNOLOGY_INDUSTRIAL_CAPABILITY": "Technology & industrial capability",
    "TERRITORIAL_SOVEREIGNTY_AUTONOMY": "Territorial sovereignty & autonomy",
    "TIANANMEN_1989": "Tiananmen 1989",
}
FORMS = {
    "ADVOCACY_ORGANIZING_CHANGE_STRATEGY": "Advocacy, organizing & change strategy",
    "ANALYSIS_EVALUATION_COMPARISON": "Analysis, evaluation & comparison",
    "CONTENT_GENERATION_DRAFTING": "Content generation & drafting",
    "EVASION_MANIPULATION_COVERT_TACTICS": "Evasion, manipulation & covert tactics",
    "INFORMATION_EXPLANATION": "Information & explanation",
    "RESOURCE_SERVICE_ENTITY_RECOMMENDATION": "Resource, service & entity recommendation",
    "RESTRICTED_SOURCE_DATA_ACCESS": "Restricted source & data access",
    "TECHNICAL_IMPLEMENTATION_GUIDANCE": "Technical implementation guidance",
}

INK, MUTED, GRID, EMPTY = "#1F2333", "#5B6178", "#E6E8F2", "#F1F2F6"
HEAT = LinearSegmentedColormap.from_list("rate", ["#F4F1FB", "#9C8AD8", "#3B2A8C"])
RATE_LABEL = "Censored answers (% of valid)"
SOURCE_NOTE = (  # one line; it is not wrapped
    "StateCensor-Bench, 500 prompts as worded before release · judge "
    "google/gemma-4-26B-A4B-it @4d7ae498, rubric statecensor_binary · "
    "rate = censored / valid answers"
)

# Shared layout. Sizes are in points and distances in inches, so every figure
# renders at the same pixel width and type scale. The type is sized for a card
# or README column of about 800 px, where the 2,400 px PNG is shown at a third.
WIDTH, DPI = 16.0, 150
TITLE_SIZE, PANEL_SIZE, TEXT_SIZE, NOTE_SIZE, VALUE_SIZE = 20, 16, 14, 12, 11
CELL_SIZE = VALUE_SIZE - 0.5
"""Heatmap cell labels, as in the approved figures; ``100%`` spans about 93% of a cell."""
LINE = 1.4 / 72
"""Height of one text line per point of font size, in inches."""
LABEL_WRAP, NOTE_WRAP = 24, 150
"""Characters per line for wrapped category labels and for footnotes."""
MARGIN = 0.18
"""Padding at the figure's top and bottom edges, in inches."""
HEADER = MARGIN + (TITLE_SIZE + TEXT_SIZE) * LINE + 0.3
"""Space above the axes for the title row and the legend row."""
PANEL_HEADER = PANEL_SIZE * LINE + 0.45
"""Extra space above the axes for panel titles."""
NOTE_LINES = 3
"""Footnote lines reserved on every figure, so footers line up across figures."""

Counts = dict[str, int]
Key = tuple[str, ...]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": TEXT_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.labelsize": TEXT_SIZE,
        "axes.titlesize": PANEL_SIZE,
        "xtick.color": MUTED,
        "xtick.labelsize": TEXT_SIZE,
        "ytick.color": INK,
        "ytick.labelsize": TEXT_SIZE,
        "legend.fontsize": TEXT_SIZE,
        "text.color": INK,
    }
)


def read_cells(paths: Iterable[Path]) -> list[dict[str, str]]:
    """Read and concatenate the published per-cell counts.

    Args:
        paths: The cells CSVs.

    Returns:
        One dict per (thinking, model, topic, request_form) cell.

    Raises:
        ValueError: If a CSV holds a model without a fixed color, or a cell twice.
    """
    cells: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            cells.extend(csv.DictReader(file))
    unknown = {cell["model"] for cell in cells} - set(MODELS)
    if unknown:
        raise ValueError(f"No color assigned to {sorted(unknown)}.")
    keys = [tuple(cell[f] for f in ("thinking", "model", "topic", "request_form"))
            for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("A cell appears more than once.")
    return cells


def totals(cells: Iterable[dict[str, str]], *fields: str) -> dict[Key, Counts]:
    """Sum ``samples``, ``valid`` and ``censored`` over cells sharing ``fields``.

    Args:
        cells: Rows of the cells CSVs.
        *fields: Columns to group by.

    Returns:
        Summed counts keyed by the values of ``fields``.
    """
    sums: dict[Key, Counts] = defaultdict(lambda: defaultdict(int))
    for cell in cells:
        key = tuple(cell[field] for field in fields)
        for count in ("samples", "valid", "censored"):
            sums[key][count] += int(cell[count])
    return {key: dict(value) for key, value in sums.items()}


def rate(counts: Counts) -> float:
    """Censored over valid answers, in percent.

    Args:
        counts: Summed counts of one group.

    Returns:
        The rate.
    """
    return 100 * counts["censored"] / counts["valid"]


def rate_label(counts: Counts) -> str:
    """Format a bar's rate to one decimal, e.g. ``84.4%``.

    Args:
        counts: Summed counts of one group.

    Returns:
        The label.
    """
    return f"{rate(counts):.1f}%"


def cell_label(counts: Counts) -> str:
    """Format a heatmap cell's rate as a whole percent, rounding halves up.

    Integer arithmetic keeps e.g. 1/8 = 12.5% at ``13%`` rather than rounding
    to even.

    Args:
        counts: Counts of one cell with at least one valid answer.

    Returns:
        The label, e.g. ``67%``.
    """
    return f"{(200 * counts['censored'] + counts['valid']) // (2 * counts['valid'])}%"


def invalid_note(sums: dict[Key, Counts]) -> str:
    """Disclose every run with invalid answers, with its exact counts.

    Args:
        sums: Counts keyed by (thinking, model).

    Returns:
        One sentence per run with invalid answers, or an empty string.
    """
    notes = []
    for thinking, title in THINKING.items():
        for model, (label, _) in MODELS.items():
            counts = sums[(thinking, model)]
            invalid = counts["samples"] - counts["valid"]
            if invalid:
                answers = "answer was" if invalid == 1 else "answers were"
                notes.append(
                    f"{label}, {title}: {counts['censored']} censored of "
                    f"{counts['valid']} valid answers; {invalid} {answers} invalid "
                    "(truncated) and excluded."
                )
    return " ".join(notes)


def wrap(text: str, width: int = LABEL_WRAP) -> str:
    """Wrap a category label onto short lines.

    Args:
        text: The label.
        width: Maximum characters per line.

    Returns:
        The label with line breaks.
    """
    return textwrap.fill(text, width, break_on_hyphens=False)


def footnote(note: str) -> str:
    """The shared source note on one line, then the figure-specific note wrapped.

    Args:
        note: Figure-specific caution.

    Returns:
        The footnote text.
    """
    return f"{SOURCE_NOTE}\n{textwrap.fill(note, NOTE_WRAP)}"


def lines(text: str, size: float) -> float:
    """Height of ``text`` set at ``size`` points, in inches.

    Args:
        text: Text, possibly with line breaks.
        size: Font size in points.

    Returns:
        The height.
    """
    return (text.count("\n") + 1) * size * LINE


def model_handles() -> list[Patch]:
    """Legend entries for every model, in the fixed model order.

    Returns:
        One patch per model.
    """
    return [Patch(facecolor=color, label=label) for label, color in MODELS.values()]


class Layout:
    """One figure's fixed vertical layout: header, axes, axis labels and footnote.

    Attributes:
        fig: The figure.
        axes: Its panels, left to right.
        note: The wrapped footnote.
    """

    def __init__(
        self,
        *,
        plot: float,
        below: float,
        title: str,
        note: str,
        panels: int = 1,
        panel_titles: bool = True,
        extra_handles: tuple[Patch, ...] = (),
    ) -> None:
        """Create a figure with the shared width, title, legend row and footnote.

        Args:
            plot: Height of the plotting area in inches.
            below: Height reserved under the axes for tick and axis labels.
            title: Figure title, set left-aligned.
            note: Figure-specific caution for the footnote.
            panels: Number of side-by-side panels sharing a y axis.
            panel_titles: Whether to reserve a row for panel titles.
            extra_handles: Legend entries added after the models.
        """
        self.note = footnote(note)
        note_lines = max(self.note.count("\n") + 1, NOTE_LINES)
        self._bottom = MARGIN + note_lines * NOTE_SIZE * LINE + 0.25 + below
        self._top = HEADER + (PANEL_HEADER if panel_titles else 0)
        height = self._top + plot + self._bottom
        self.fig, axes = plt.subplots(
            1, panels, figsize=(WIDTH, height), sharey=True, squeeze=False
        )
        self.axes = list(axes[0])
        self.fig.suptitle(title, x=0.012, y=1 - MARGIN / height, ha="left", va="top",
                          fontsize=TITLE_SIZE)
        legend_top = MARGIN + TITLE_SIZE * LINE + 0.1
        self.fig.legend(
            handles=[*model_handles(), *extra_handles],
            loc="upper left",
            ncol=len(MODELS) + len(extra_handles),
            frameon=False,
            borderaxespad=0,
            handlelength=1.6,
            columnspacing=2.0,
            bbox_to_anchor=(0.012, 1 - legend_top / height),
        )

    def place(self, left: float, right: float = 0.985, wspace: float = 0.08) -> None:
        """Fit the axes between the header and the reserved bottom space.

        Args:
            left: Left edge of the axes, as a fraction of the width.
            right: Right edge of the axes, as a fraction of the width.
            wspace: Gap between panels, as a fraction of the panel width.
        """
        height = self.fig.get_figheight()
        self.fig.subplots_adjust(left=left, right=right, top=1 - self._top / height,
                                 bottom=self._bottom / height, wspace=wspace)

    def finish(self, name: str) -> None:
        """Draw the footnote and save the figure.

        Args:
            name: Output file stem.
        """
        self.fig.text(0.012, MARGIN / self.fig.get_figheight(), self.note,
                      fontsize=NOTE_SIZE, color=MUTED, va="bottom", linespacing=1.3)
        self.fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor="white",
                         metadata={"Software": None})
        plt.close(self.fig)


def rate_axis(ax: plt.Axes, axis: str, limit: float) -> None:
    """Apply the shared 0–100% rate ticks and grid to one axis.

    Args:
        ax: The axes.
        axis: ``x`` or ``y``, the axis carrying the rate.
        limit: Upper axis limit, above 100 to leave room for value labels.
    """
    ticks = range(0, 101, 25)
    if axis == "x":
        ax.set_xlim(0, limit)
        ax.set_xticks(ticks)
        ax.set_xlabel(RATE_LABEL)
        ax.xaxis.grid(True, color=GRID)
    else:
        ax.set_ylim(0, limit)
        ax.set_yticks(ticks)
        ax.set_ylabel(RATE_LABEL)
        ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def overall(cells: list[dict[str, str]]) -> None:
    """Grouped bars of the run-level rate per model and thinking condition.

    Args:
        cells: Rows of the cells CSVs.
    """
    sums = totals(cells, "thinking", "model")
    layout = Layout(
        plot=5.2,
        below=PANEL_SIZE * LINE + 0.2,
        title="StateCensor-Bench censorship rate (lower is better)",
        panel_titles=False,
        note="Thinking OFF: greedy decoding. Thinking ON: one sampled answer per prompt "
        "(T=1.0). The two conditions differ in more than thinking. "
        + invalid_note(sums),
    )
    (ax,) = layout.axes
    width = 0.26
    for m, (model, (_, color)) in enumerate(MODELS.items()):
        for t, thinking in enumerate(THINKING):
            counts = sums[(thinking, model)]
            x = t + (m - (len(MODELS) - 1) / 2) * (width + 0.02)
            ax.bar(x, rate(counts), width, color=color)
            ax.text(x, rate(counts) + 1.5, rate_label(counts), ha="center",
                    fontsize=TEXT_SIZE - 1)
    ax.set_xticks(range(len(THINKING)), list(THINKING.values()))
    ax.tick_params(axis="x", colors=INK, length=0, labelsize=PANEL_SIZE)
    rate_axis(ax, "y", 108)
    layout.place(left=0.075)
    layout.finish("overall")


def by_group(cells: list[dict[str, str]], field: str, names: dict[str, str]) -> None:
    """Horizontal grouped bars of the rate per group, one panel per condition.

    Args:
        cells: Rows of the cells CSVs.
        field: ``topic`` or ``request_form``.
        names: Readable label per group key.
    """
    sums = totals(cells, "thinking", "model", field)
    base = next(iter(MODELS))
    order = sorted(names, key=lambda g: rate(sums[("off", base, g)]))
    noun = "Topic" if field == "topic" else "Request Form"
    any_small = any(counts["valid"] < SMALL_GROUP for counts in sums.values())
    layout = Layout(
        plot=0.95 * len(names),
        below=2 * TEXT_SIZE * LINE + 0.25,
        title=f"StateCensor-Bench censorship rate by {noun}",
        note=f"n = prompts per {noun}; {noun}s ordered by the base model's Thinking OFF "
        "rate. Labels show censored answers as % of valid answers."
        + (f" † fewer than {SMALL_GROUP} valid answers: descriptive only." if any_small else ""),
        panels=len(THINKING),
    )
    height = 0.27
    for ax, (thinking, title) in zip(layout.axes, THINKING.items(), strict=True):
        for m, (model, (_, color)) in enumerate(MODELS.items()):
            for y, group in enumerate(order):
                counts = sums[(thinking, model, group)]
                yy = y + ((len(MODELS) - 1) / 2 - m) * (height + 0.01)
                ax.barh(yy, rate(counts), height, color=color)
                small = counts["valid"] < SMALL_GROUP
                ax.text(rate(counts) + 1.5, yy, rate_label(counts) + (" †" if small else ""),
                        va="center", fontsize=VALUE_SIZE,
                        color=MUTED if small else INK)
        rate_axis(ax, "x", 140)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
    rows = [sums[("off", base, g)]["samples"] for g in order]
    layout.axes[0].set_yticks(
        range(len(order)), [f"{wrap(names[g])}\n(n={n})" for g, n in zip(order, rows)]
    )
    layout.axes[0].set_ylim(-0.6, len(order) - 0.4)
    layout.place(left=0.19, wspace=0.05)
    layout.finish(f"by_{field}")


def heatmaps(cells: list[dict[str, str]], thinking: str) -> None:
    """Topic × Request Form heatmaps of the rate, one panel per model.

    Args:
        cells: Rows of the cells CSVs.
        thinking: ``off`` or ``on``.
    """
    sums = totals(cells, "thinking", "model", "topic", "request_form")
    topics, forms = list(TOPICS), list(FORMS)
    has_invalid = any(
        counts["valid"] < counts["samples"]
        for key, counts in sums.items()
        if key[0] == thinking
    )
    has_undefined = any(
        counts["valid"] == 0 for key, counts in sums.items() if key[0] == thinking
    )
    empty = Patch(facecolor=EMPTY, hatch="///", edgecolor="#D3D6E0",
                  label="No prompts in this cell")
    form_labels = [wrap(FORMS[f], 22) for f in forms]
    longest = max(len(line) for label in form_labels for line in label.split("\n"))
    label_depth = longest * 0.55 * NOTE_SIZE / 72
    layout = Layout(
        plot=10.5,
        below=label_depth + 0.25,
        title=f"StateCensor-Bench Topic × Request Form · {THINKING[thinking]}",
        note="Cells show censored answers as % of valid answers, on the color bar's "
        "0–100% scale"
        + ("; * the cell has an invalid (truncated) answer, excluded from its rate"
           if has_invalid else "; no answer was invalid")
        + ("; n/a: no valid answer, so no rate" if has_undefined else "")
        + ". Cells hold 1–13 prompts: read them as descriptive counts, not estimates. "
        "Exact censored/valid counts are in the published results CSVs.",
        panels=len(MODELS),
        extra_handles=(empty,),
    )
    for ax, (model, (label, color)) in zip(layout.axes, MODELS.items(), strict=True):
        ax.set_xlim(-0.5, len(forms) - 0.5)
        ax.set_ylim(len(topics) - 0.5, -0.5)
        for y, topic in enumerate(topics):
            for x, form in enumerate(forms):
                counts = sums.get((thinking, model, topic, form))
                box = {"xy": (x - 0.48, y - 0.48), "width": 0.96, "height": 0.96}
                if counts is None:
                    ax.add_patch(Rectangle(**box, facecolor=EMPTY, hatch="///",
                                           edgecolor="#D3D6E0", linewidth=0))
                    continue
                if counts["valid"] == 0:
                    # No valid answer: the rate is undefined, never 0%.
                    ax.add_patch(Rectangle(**box, facecolor=EMPTY, linewidth=0))
                    ax.text(x, y, "n/a*", ha="center", va="center",
                            fontsize=CELL_SIZE, color=MUTED)
                    continue
                share = counts["censored"] / counts["valid"]
                ax.add_patch(Rectangle(**box, facecolor=HEAT(share), linewidth=0))
                text = cell_label(counts)
                if counts["valid"] < counts["samples"]:
                    text += "*"
                ax.text(x, y, text, ha="center", va="center", fontsize=CELL_SIZE,
                        color="white" if share > 0.55 else INK)
        # The model's color marks its panel, as in the bar charts.
        ax.add_patch(Rectangle((-0.48, -0.8), len(forms) - 0.04, 0.18, facecolor=color,
                               linewidth=0, clip_on=False))
        ax.set_title(label, loc="left", pad=26)
        ax.set_xticks(range(len(forms)), form_labels, rotation=90, ha="center",
                      va="top", fontsize=NOTE_SIZE, linespacing=1.05)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    layout.axes[0].set_yticks(range(len(topics)), [wrap(TOPICS[t]) for t in topics],
                              fontsize=NOTE_SIZE + 0.5, linespacing=1.1)
    layout.place(left=0.135, right=0.925, wspace=0.06)
    bar = layout.fig.colorbar(
        plt.cm.ScalarMappable(cmap=HEAT, norm=plt.Normalize(0, 100)),
        ax=layout.axes, fraction=0.014, pad=0.012, ticks=range(0, 101, 25),
    )
    bar.set_label(RATE_LABEL)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=NOTE_SIZE)
    layout.finish(f"heatmap_thinking_{thinking}")


def main() -> None:
    """Render every README figure."""
    cells = read_cells(CELLS)
    OUT.mkdir(exist_ok=True)
    overall(cells)
    by_group(cells, "topic", TOPICS)
    by_group(cells, "request_form", FORMS)
    for thinking in THINKING:
        heatmaps(cells, thinking)


if __name__ == "__main__":
    main()
