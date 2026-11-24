"""Figures for the paper, one function per figure, written to figures/ as vector PDF.

Usage: uv run python figures.py [fig1 ...]   (no arguments: every figure)

Figures 1 to 5 of the paper are fig1, fig2, fig3_v2, fig4 and fig5; the supplementary figures are figS_humanset, figS_forecasters30 and figS_ablation30. Every plotted value is read from the cached outputs through analysis.py and checked against the value reported in the paper; a mismatch stops the script.
"""

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

import analysis as A
import score as S

OUT = S.ROOT / "figures"
FONT = "Liberation Sans"  # metric-compatible with Arial; embedded in the PDF as TrueType
TOKEN = "#0072B2"   # token score (Okabe-Ito blue), throughout the paper
STATED = "#E69F00"  # stated score (Okabe-Ito orange), throughout the paper
HUMAN = "#6E6E6E"   # human forecasters
INK = "#1A1A1A"
MUTED = "#767676"   # fixed template wording, secondary labels
RULE = "#9A9A9A"    # box outlines
TINT = "#F1F1F1"    # background of the text filled in from the claim's row
DOUBLE_COLUMN_MM = 183
MM_PER_PT = 25.4 / 72

plt.rcParams.update({
    "font.family": FONT, "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.5, "axes.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "lines.linewidth": 0.8,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic", "mathtext.bf": FONT + ":bold",
})


class FigureError(Exception):
    """A plotted value does not match the value expected from the write-up."""


def check(what: str, found, expected, places: int = 3) -> None:
    same = round(found, places) == round(expected, places) if isinstance(expected, float) else found == expected
    if not same:
        raise FigureError(f"{what}: found {found!r}, expected {expected!r}")


def canvas(width_mm: float, height_mm: float):
    """A figure of the given size whose single axes is measured in millimetres from the bottom left, with nothing drawn."""
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.axis("off")
    return fig, ax


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def panel(fig, left: float, bottom: float, width: float, height: float):
    """Axes placed in millimetres from the bottom left of the figure."""
    w, h = fig.get_size_inches() * 25.4
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def panel_label(fig, x: float, letter: str) -> None:
    """A bold lowercase panel label at the top of the figure, x millimetres from the left."""
    w, h = fig.get_size_inches() * 25.4
    fig.text(x / w, 1 - 0.8 / h, letter, size=8, weight="bold", va="top", ha="left", color=INK)


def auc_axis(ax, lo: float = 0.4, hi: float = 1.0, top: float | None = None) -> None:
    """The shared AUC axis: ticks from lo to hi, a dashed chance line at 0.5, and room above hi (up to top) for brackets."""
    ax.set_ylim(lo, top or hi)
    ax.set_yticks(np.arange(round(lo * 10), round(hi * 10) + 1) / 10)
    ax.spines["left"].set_bounds(lo, hi)
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (4, 3)), zorder=0)
    ax.set_ylabel("AUC")


def interval(v) -> tuple[float, float]:
    """The 95% percentile interval of bootstrap values."""
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def bracket(ax, x0: float, x1: float, y: float, label: str, drop: float = 0.012, color: str = INK) -> None:
    """A bracket joining two x positions at height y, with its label centred above."""
    ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop], color=color, lw=0.5, solid_capstyle="butt", solid_joinstyle="miter", clip_on=False, zorder=3)
    ax.text((x0 + x1) / 2, y + 0.004, label, size=5.5, ha="center", va="bottom", color=INK)


def signed(v: float) -> str:
    """A number with its sign and three decimals, with a true minus sign."""
    return f"{v:+.3f}".replace("-", "\u2212")


def human_bands(ax, x: float, aggregates: bool = True, ha: str = "right") -> None:
    """Light grey reference bands on an AUC axis of the high-power set, labelled at x (the first only if aggregates is true): the range of the two forecaster aggregates (repliCATS group mean, Replication Markets final price) and the range of the two typical individuals (Supplementary Sections S9 and S11)."""
    rows = A.sel("highpower")
    crowd = [A.auc(h, rows) for h in A.humans(rows)]
    est = A.individual_estimates()
    individuals = [float(A.typical_fixed([r for r in rows if r.row in est[src]], est[src]).mean()) for src in A.INDIVIDUAL_SOURCES]
    for what, found, e in zip(("repliCATS group mean", "Replication Markets final price", "typical repliCATS individual", "typical Markets survey individual"), crowd + individuals, (0.731, 0.723, 0.680, 0.675)):
        check(f"reference band, {what}", found, e)
    for values, label, va in ((crowd, "Forecaster aggregates", "bottom"), (individuals, "Typical individuals", "top")):
        if values is crowd and not aggregates:
            continue
        lo, hi = min(values), max(values)
        ax.axhspan(lo, hi, color="#BDBDBD", lw=0, zorder=0)
        ax.text(x, hi + 0.006 if va == "bottom" else lo - 0.006, label, size=5.5, color=HUMAN, ha=ha, va=va)


def text_width(s: str, size: float, weight: str = "normal") -> float:
    """Width in millimetres of a string at the given point size."""
    if not s.strip():
        return 0.0
    return TextPath((0, 0), s, size=size, prop=FontProperties(family=FONT, weight=weight)).get_extents().width * MM_PER_PT


def wrap(s: str, size: float, width: float) -> list[str]:
    """The string broken into lines no wider than width millimetres; line breaks already in the string are kept."""
    lines = []
    for piece in s.split("\n"):
        line = ""
        for word in piece.split():
            trial = f"{line} {word}".strip()
            if line and text_width(trial, size) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def box(ax, x0, y0, x1, y1, face="white", edge=RULE, lw=0.6, dashed=False, radius=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={radius}", fc=face, ec=edge, lw=lw, ls=(0, (3, 2)) if dashed else "-", zorder=z))


def arrow(ax, points, color=INK, lw=0.7, dashed=False):
    """A path through the points (straight legs), ending in an arrowhead."""
    ls = (0, (3, 2)) if dashed else "-"
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=3)
    ax.annotate("", xy=points[-1], xytext=points[-2], zorder=3, arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=6))


def answer_word_probabilities(e: dict) -> tuple[float, float]:
    """p(yes) and p(no) at the answer word of a cached response, with spellings of the same word summed as in score.token_probability."""
    p = {"yes": 0.0, "no": 0.0}
    for tok, lp in e["answer_top_logprobs"]:
        if tok.strip().lower() in p:
            p[tok.strip().lower()] += math.exp(lp)
    return p["yes"], p["no"]


# ---------------------------------------------------------------- Figure 1


def fig1():
    """Schematic of the task and the two readouts, filled with the worked example of Supplementary Section S4 (row 239) and Qwen3.8's cached response to it without reasoning."""
    rec = A.BY_ROW[239]
    e = A.qwen(rec, A.F, "off")
    p_yes, p_no = answer_word_probabilities(e)
    token, stated = A.token(e), e["probability"] / 100
    check("example paper", rec.paper.startswith("Woltin"), True)
    check("example in the high-power set", rec.highpower, True)
    check("example response", e["content"], "Answer: no\nProbability: 15")
    check("example token score", token, 0.020)
    check("example p(yes) relative to p(no)", p_yes / (p_yes + p_no), token)
    check("example stated score", stated, 0.15)

    intro, abstract, claim_intro, claim, finding_intro, finding, question = S._template_parts(rec, A.F)
    cut = " (i.e.,"
    check("abstract contains the cut point", cut in abstract, True)
    paragraphs = [  # (text, label of the part filled in from the claim's row, or None for fixed template wording)
        (intro, None), (abstract.split(cut)[0] + " ...", "Abstract"), (claim_intro, None), (claim, "Claim tested"),
        (finding_intro, None), (finding, "Original finding"), (question, None), (S.QWEN_INSTRUCTION, None),
    ]

    # layout, in millimetres
    W = DOUBLE_COLUMN_MM
    size, lead, gap, pad = 5.5, 2.35, 1.25, 2.0          # prompt type size (points), line spacing, paragraph spacing, card padding
    head = 5.0                                            # room for the headings above the boxes
    card = (0.4, 79.0)                                    # prompt card, left and right edges
    text_x = (16.0, card[1] - pad)                        # prompt text; the part labels sit to its left
    laid = [(wrap(t, size, text_x[1] - text_x[0] - 0.8), label) for t, label in paragraphs]
    card_h = 2 * pad + sum(len(lines) * lead for lines, _ in laid) + gap * (len(laid) - 1)
    H = card_h + head + 0.4
    top = H - head
    fig, ax = canvas(W, H)

    def heading(x, s, above=None):
        """A bold heading at the top of the figure, or just above a box whose top is given."""
        ax.text(x, H - 0.6 if above is None else above + 3.6, s, size=7, weight="bold", color=INK, va="top", ha="left")

    # left: the prompt
    heading(card[0], "Prompt")
    ax.text(card[1], H - 0.9, f"fixed template, filled with one claim (example: {rec.paper.replace(' (', ', ').rstrip(')')})", size=5, color=MUTED, va="top", ha="right")
    box(ax, card[0], top - card_h, card[1], top)
    y = top - pad
    for lines, label in laid:
        h = len(lines) * lead
        if label:
            ax.add_patch(Rectangle((text_x[0] - 0.8, y - h - 0.35), text_x[1] - text_x[0] + 1.6, h + 0.7, fc=TINT, ec="none", zorder=2))
            ax.text(card[0] + pad, y - 0.15, label.replace(" ", "\n"), size=size, weight="bold", color=INK, va="top", ha="left", linespacing=1.15, zorder=4)
        for i, line in enumerate(lines):
            ax.text(text_x[0], y - i * lead - 0.15, line, size=size, color=INK if label else MUTED, va="top", ha="left", zorder=4)
        y -= h + gap

    # middle: the model, with reasoning off (main path) or on (secondary path)
    block = (top - card_h, top)                           # vertical extent of the model, response and readouts
    yc = block[1] - 19.0                                  # height of the main path
    model = (86.0, 108.0)
    resp = (124.0, 147.5)
    heading(model[0], "Model", above=yc + 6.5)
    box(ax, model[0], yc - 6.5, model[1], yc + 6.5, face="#FAFAFA", edge=INK, lw=0.8)
    ax.text(sum(model) / 2, yc + 1.4, "Language", size=6.5, weight="bold", ha="center", va="center", color=INK)
    ax.text(sum(model) / 2, yc - 1.4, "model", size=6.5, weight="bold", ha="center", va="center", color=INK)
    arrow(ax, [(card[1], yc), (model[0], yc)])
    arrow(ax, [(model[1], yc), (resp[0], yc)])
    mid = (model[1] + resp[0]) / 2
    ax.text(mid - 0.4, yc + 1.0, "Reasoning off", size=5.5, weight="bold", ha="center", va="bottom", color=INK)
    ax.text(mid - 0.4, yc - 1.0, "single\nforward pass", size=5.5, ha="center", va="top", color=INK, linespacing=1.15)
    reason = (yc - 28.0, yc - 17.0)                       # reasoning box, bottom and top
    box(ax, model[0], reason[0], model[1] + 6.0, reason[1], edge=MUTED, dashed=True)
    ax.text(model[0] + 14.0, sum(reason) / 2 + 2.6, "Reasoning on", size=5.5, weight="bold", ha="center", va="center", color=MUTED)
    ax.text(model[0] + 14.0, sum(reason) / 2 - 1.3, "the model writes its\nreasoning, then answers", size=5.5, ha="center", va="center", color=MUTED, linespacing=1.15)
    arrow(ax, [(sum(model) / 2, yc - 6.5), (sum(model) / 2, reason[1])], color=MUTED, dashed=True)
    arrow(ax, [(model[1] + 6.0, sum(reason) / 2), (resp[0] + 6.0, sum(reason) / 2), (resp[0] + 6.0, yc - 6.5)], color=MUTED, dashed=True)

    # right: the response and the two readouts taken from it
    heading(resp[0], "Response", above=yc + 6.5)
    box(ax, resp[0], yc - 6.5, resp[1], yc + 6.5)
    rsize, rx = 6.5, resp[0] + 2.2
    answer_line, prob_line = e["content"].split("\n")
    lines = ((answer_line, yc + 2.3, TOKEN, "token"), (prob_line, yc - 2.3, STATED, "stated"))
    wx = rx + max(text_width(line.rsplit(" ", 1)[0], rsize) for line, *_ in lines) + 1.6  # both highlighted words start here, so they align
    marks = {}
    for line, ly, colour, name in lines:
        stem, word = line.rsplit(" ", 1)
        ww = text_width(word, rsize)
        ax.add_patch(FancyBboxPatch((wx - 0.7, ly - 1.75), ww + 1.4, 3.5, boxstyle="round,pad=0,rounding_size=0.6", fc=colour + "30", ec=colour, lw=0.7, zorder=2))
        ax.text(rx, ly, stem, size=rsize, color=INK, va="center_baseline", ha="left", zorder=4)
        ax.text(wx, ly, word, size=rsize, weight="bold", color=INK, va="center_baseline", ha="left", zorder=4)
        marks[name] = (wx + ww + 0.7, ly)

    out = (153.0, W - 0.4)                                # readout panels, left and right edges
    heading(out[0], "Readouts")
    tok = (block[1] - 33.0, block[1])                     # token score panel, bottom and top
    sta = (block[0], tok[0] - 4.0)                        # stated score panel
    elbow = (resp[1] + out[0]) / 2 + 0.3
    for name, colour, (y0, y1), title, note in (("token", TOKEN, tok, "Token score", "read at the answer word"), ("stated", STATED, sta, "Stated score", "the probability the model writes")):
        box(ax, out[0], y0, out[1], y1, edge=colour, lw=0.9)
        ax.text(out[0] + 2.0, y1 - 1.6, title, size=6.5, weight="bold", color=INK, va="top", ha="left")
        ax.text(out[0] + 2.0, y1 - 4.5, note, size=5, color=MUTED, va="top", ha="left")
        mx, my = marks[name]
        target = y1 - 2.9
        arrow(ax, [(mx, my), (elbow, my), (elbow, target), (out[0], target)], color=colour, lw=0.9)

    # token score: p(yes) and p(no) at the answer word, and their ratio
    bx, full = out[0] + 9.5, 12.5                         # bars start here; length of a bar for probability 1
    for i, (word, p) in enumerate((("yes", p_yes), ("no", p_no))):
        by = tok[1] - 12.5 - i * 4.0
        ax.text(bx - 1.0, by, f"p({word})", size=5.5, color=INK, va="center_baseline", ha="right")
        ax.add_patch(Rectangle((bx, by - 1.2), full, 2.4, fc="#EDEDED", ec="none", zorder=2))
        ax.add_patch(Rectangle((bx, by - 1.2), full * p, 2.4, fc=TOKEN, ec="none", zorder=3))
        ax.text(bx + full + 1.0, by, f"{p:.3f}", size=5.5, color=INK, va="center_baseline", ha="left")
    fy = tok[0] + 6.5
    ax.text(out[0] + 2.0, fy, r"$\dfrac{p(\mathrm{yes})}{p(\mathrm{yes}) + p(\mathrm{no})}$", size=6, color=INK, va="center", ha="left")
    ax.text(out[1] - 2.0, fy, f"= {token:.3f}", size=6.5, weight="bold", color=INK, va="center_baseline", ha="right")

    # stated score: the number the model writes, divided by 100
    sy = (sta[0] + sta[1] - 6.5) / 2
    ax.text(out[0] + 2.0, sy, f"{e['probability']} / 100", size=6, color=INK, va="center_baseline", ha="left")
    ax.text(out[1] - 2.0, sy, f"= {stated:.2f}", size=6.5, weight="bold", color=INK, va="center_baseline", ha="right")

    save(fig, "fig1")


# ---------------------------------------------------------------- Figure 2

QWEN30I = A.PANEL_MODELS["Qwen3 30B-A3B Instruct"]


def forecaster_panel(fig, bx, rows, model: str, name: str, expected: dict, weighted_label: bool) -> None:
    """A model (a key of analysis.PANEL_MODELS) against each qualifying Replication Markets survey forecaster on that forecaster's claims among rows, drawn in the axes bx: the forecaster's AUC against the model's AUC on the same claims, marker area by success-failure pairs. Labelled with the mean advantage weighted by pairs or, if weighted_label is false, with the share of forecasters outperformed and the equally weighted mean advantage. Statistics as in Supplementary Section S10 (analysis.py individual_tests)."""
    short = model.replace(" Instruct", "").replace(" 27B", "")
    est = A.individual_estimates()
    kept = [r for r in rows if r.row in est["RM_survey"]]
    fc = A._forecasters(kept, A.survey_estimates())
    t = A.forecaster_test(kept, fc, A.chat_scores(kept, A.PANEL_MODELS[model])[0])
    fa, ma, pairs = t["forecaster_auc"], t["model_auc"], t["pairs"]
    weighted = (t["point"][2], *interval(t["boot"][:, 2]), A._boot_p(t["boot"][:, 2]))
    equal = (t["point"][1], *interval(t["boot"][:, 1]), A._boot_p(t["boot"][:, 1]))
    share = (t["point"][0], *(0.5 + v for v in interval(t["boot"][:, 0])), A._boot_p(t["boot"][:, 0]))  # the bootstrap holds the share's excess over 0.5
    found = {"forecasters": len(fc), "share": float(share[0]), "share lower": share[1], "share upper": share[2], "share p": share[3], "forecaster mean": float(fa.mean()), "model mean": float(ma.mean()),
             "weighted": float(weighted[0]), "weighted lower": weighted[1], "weighted upper": weighted[2], "weighted p": weighted[3],
             "equal": float(equal[0]), "equal lower": equal[1], "equal upper": equal[2], "equal p": equal[3]}
    for k, e in expected.items():
        check(f"{name} {k}", found[k], e, places=2 if k in ("share lower", "share upper") else 3)
    scale = 106 / pairs.max()  # marker area in square points per success-failure pair; the forecaster with the most pairs gets the same marker in every figure
    bx.plot([0, 1], [0, 1], color=MUTED, lw=0.5, ls=(0, (4, 3)), zorder=1)
    bx.scatter(fa, ma, s=pairs * scale, fc=TOKEN, ec=TOKEN, alpha=0.45, lw=0.5, zorder=3, clip_on=False)
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.set_aspect("equal")
    bx.set_xticks(np.arange(0, 1.01, 0.2))
    bx.set_yticks(np.arange(0, 1.01, 0.2))
    bx.set_xlabel("Forecaster's AUC")
    bx.set_ylabel(f"{model} AUC on the same claims")
    bx.text(0.04, 0.965, f"{short} better", size=5.5, color=INK, va="top", ha="left")
    bx.text(0.97, 0.03, "forecaster better", size=5.5, color=INK, va="bottom", ha="right")
    if weighted_label:
        label = f"Mean advantage, weighted by pairs:\n{signed(weighted[0])} (95% CI {signed(weighted[1])} to {signed(weighted[2])})\n{p_text(weighted[3])}"
    else:
        label = (f"Outperforms {share[0]:.0%} of {len(fc)} forecasters\n(95% CI {share[1]:.0%} to {share[2]:.0%}), {p_text(share[3])}\n"
                 f"Mean advantage: {signed(equal[0])}\n(95% CI {signed(equal[1])} to {signed(equal[2])}), {p_text(equal[3])}")
    bx.text(0.97, 0.085, label, size=5.5, color=INK, va="bottom", ha="right", linespacing=1.25)
    sizes = [n for n in (10, 100, 400, 1000, 2500) if n <= pairs.max()][-3:]
    handles = [bx.scatter([], [], s=n * scale, fc=TOKEN, ec=TOKEN, alpha=0.45, lw=0.5) for n in sizes]
    fig.legend(handles, [str(n) for n in sizes], title="Success-failure pairs", loc="upper right", bbox_to_anchor=(0.985, 1.0), ncols=len(sizes), frameon=False, fontsize=5.5, title_fontsize=5.5,
               handletextpad=0.2, columnspacing=1.0, borderpad=0.2)


def humans_figure(subset: str, name: str, expected: dict, weighted_label: bool = True) -> None:
    """Models against human forecasters on one set (Figure 2: high-power set; supplementary figure: human set). Panel a: AUC of the two models' token scores, the typical individual of each team and the two aggregates, with brackets for model against typical individual. Panel b: Qwen3.8 against each survey forecaster on that forecaster's claims, labelled with the mean advantage weighted by pairs or, if weighted_label is false, with the share of forecasters outperformed and the equally weighted mean advantage. Statistics as in Supplementary Sections S9 to S11."""
    rows = A.sel(subset)
    models = {"Qwen3.8 27B": A.chat_scores(rows, A.QWEN)[0], "Qwen3 30B-A3B Instruct": A.chat_scores(rows, QWEN30I)[0]}
    scores = {**models, **dict(zip(("repliCATS group mean", "Replication Markets final price"), A.humans(rows)))}
    boot = A.boot_aucs(scores, rows)
    points = {k: (A.auc(v, rows), *interval(boot[k])) for k, v in scores.items()}

    # typical individuals: AUC from rater draws on the claims as they are; interval with claims and raters resampled; tests with the individual averaged over raters
    est = A.individual_estimates()
    tests = {}
    for src, label in (("repliCATS_R1", "repliCATS"), ("RM_survey", "Replication Markets survey")):
        kept = [r for r in rows if r.row in est[src]]
        xm = {m: A.chat_scores(kept, mid)[0] for m, mid in A.PANEL_MODELS.items()}
        typical = A.typical_fixed(kept, est[src]).mean()
        points[label] = (typical, *interval(A.typical_single(kept, est[src], {})[0]))
        avg, bm = A.typical_averaged(kept, est[src], xm)
        for m in xm:
            d = bm[m] - avg
            tests[m, label] = (A.auc(xm[m], kept) - typical, *interval(d), A._boot_p(d))
    for k, want in expected["points"].items():
        for what, found, e in zip(("AUC", "lower", "upper"), points[k], want):
            check(f"{name}a {k} {what}", found, e)
    for k, want in expected["tests"].items():
        for what, found, e in zip(("difference", "lower", "upper", "p"), tests[k], want):
            if e is not None:
                check(f"{name}a {k[0]} minus {k[1]} individual {what}", found, e)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_MM / 25.4, 76 / 25.4))
    forecaster_panel(fig, panel(fig, 127, 12, 52, 52), rows, "Qwen3.8 27B", name + "b", expected["forecasters"], weighted_label)
    panel_label(fig, 116, "b")

    # panel a
    ax = panel(fig, 11, 21, 101, 52)
    panel_label(fig, 0.5, "a")
    order = ("Qwen3.8 27B", "Qwen3 30B-A3B Instruct", "repliCATS", "Replication Markets survey", "repliCATS group mean", "Replication Markets final price")
    ticks = ("Qwen3.8\n27B", "Qwen3 30B-A3B\nInstruct", "repliCATS\n(Round 1)", "Replication\nMarkets survey", "repliCATS\ngroup mean", "Replication Markets\nfinal price")
    marks = (dict(marker="o", mfc=TOKEN, mec=TOKEN), dict(marker="o", mfc="white", mec=TOKEN), dict(marker="s", mfc=HUMAN, mec=HUMAN), dict(marker="s", mfc=HUMAN, mec=HUMAN),
             dict(marker="D", mfc="white", mec=HUMAN), dict(marker="D", mfc="white", mec=HUMAN))
    for x, (k, mk) in enumerate(zip(order, marks)):
        a, lo, hi = points[k]
        ax.errorbar(x, a, yerr=[[a - lo], [hi - a]], color=mk["mec"], elinewidth=0.9, ms=4.6 if mk["marker"] == "o" else 4.0, mew=0.9, ls="none", zorder=3, **mk)
        ax.text(x + 0.13, a, f"{a:.3f}", size=5.5, va="center", ha="left", color=INK)
    auc_axis(ax, top=1.17)
    ax.set_xlim(-0.55, 5.6)
    ax.set_xticks(range(6), ticks)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0.4, color=INK, lw=0.5, zorder=1)
    for x0, x1, s in ((0, 1, "Language model, token score"), (2, 3, "Typical individual forecaster"), (4, 5, "Forecaster aggregate")):
        ax.plot([x0 - 0.42, x1 + 0.42], [0.262, 0.262], color=MUTED, lw=0.5, clip_on=False)
        ax.text((x0 + x1) / 2, 0.25, s, size=6, weight="bold", ha="center", va="top", color=INK)
    x_of = {k: i for i, k in enumerate(order)}
    levels = ((("Qwen3 30B-A3B Instruct", "repliCATS"), 0.07, -0.07), (("Qwen3 30B-A3B Instruct", "Replication Markets survey"), -0.07, -0.07),
              (("Qwen3.8 27B", "repliCATS"), 0.07, 0.07), (("Qwen3.8 27B", "Replication Markets survey"), -0.07, 0.07))
    base = max(hi for _, _, hi in points.values()) + 0.033
    for i, (k, d0, d1) in enumerate(levels):
        bracket(ax, x_of[k[0]] + d0, x_of[k[1]] + d1, base + 0.05 * i, p_text(tests[k][3]))

    save(fig, name)


def fig2():
    """Models against human forecasters on the 57 high-power claims."""
    humans_figure("highpower", "fig2", {
        "points": {"Qwen3.8 27B": (0.838, 0.723, 0.932), "Qwen3 30B-A3B Instruct": (0.816, 0.695, 0.912), "repliCATS": (0.680, 0.533, 0.811),
                   "Replication Markets survey": (0.675, 0.529, 0.813), "repliCATS group mean": (0.731, 0.592, 0.856), "Replication Markets final price": (0.723, 0.580, 0.849)},
        "tests": {("Qwen3.8 27B", "repliCATS"): (0.158, 0.052, 0.263, 0.005), ("Qwen3.8 27B", "Replication Markets survey"): (0.170, 0.058, 0.271, 0.005),
                  ("Qwen3 30B-A3B Instruct", "repliCATS"): (0.136, 0.018, 0.246, 0.028), ("Qwen3 30B-A3B Instruct", "Replication Markets survey"): (0.155, 0.054, 0.248, 0.003)},
        "forecasters": {"forecasters": 41, "share": 0.732, "forecaster mean": 0.577, "model mean": 0.703, "weighted": 0.150, "weighted lower": 0.011, "weighted upper": 0.287, "weighted p": 0.034},
    })


def figS_forecasters30():
    """Supplementary figure: Qwen3 30B-A3B Instruct, released before the outcomes were public, against each survey forecaster on the high-power set."""
    fig = plt.figure(figsize=(89 / 25.4, 72 / 25.4))
    forecaster_panel(fig, panel(fig, 22, 12, 52, 52), A.sel("highpower"), "Qwen3 30B-A3B Instruct", "figS_forecasters30", {
        "forecasters": 41, "share": 0.878, "share lower": 0.61, "share upper": 0.96, "share p": 0.008, "forecaster mean": 0.577, "model mean": 0.777,
        "equal": 0.200, "equal lower": 0.050, "equal upper": 0.361, "equal p": 0.014, "weighted": 0.204, "weighted lower": 0.075, "weighted upper": 0.334, "weighted p": 0.003}, weighted_label=False)
    save(fig, "figS_forecasters30")


def figS_humanset():
    """Supplementary figure: models against human forecasters on the 135 claims of the human set."""
    humans_figure("human", "figS_humanset", {
        "points": {"Qwen3.8 27B": (0.691, 0.603, 0.776), "Qwen3 30B-A3B Instruct": (0.659, 0.567, 0.751), "repliCATS": (0.594, 0.499, 0.689),
                   "Replication Markets survey": (0.587, 0.490, 0.680), "repliCATS group mean": (0.618, 0.521, 0.711), "Replication Markets final price": (0.634, 0.540, 0.730)},
        "tests": {("Qwen3.8 27B", "repliCATS"): (0.096, 0.015, 0.178, 0.022), ("Qwen3.8 27B", "Replication Markets survey"): (0.105, 0.034, 0.175, 0.004),
                  ("Qwen3 30B-A3B Instruct", "repliCATS"): (0.065, -0.025, 0.154, None), ("Qwen3 30B-A3B Instruct", "Replication Markets survey"): (0.078, -0.011, 0.165, None)},
        "forecasters": {"forecasters": 91, "share": 0.742, "share lower": 0.52, "share upper": 0.85, "share p": 0.037, "forecaster mean": 0.566, "model mean": 0.704,
                        "equal": 0.138, "equal lower": 0.023, "equal upper": 0.246, "equal p": 0.020, "weighted": 0.115, "weighted lower": 0.029, "weighted upper": 0.202},
    }, weighted_label=False)


# ---------------------------------------------------------------- Figure 3

TOKEN_LIGHT = "#7FB8DA"  # the token score's blue, lighter, for the second line of a panel


def ablation_figure(model: str, name: str, expected: dict, interaction: bool = False) -> None:
    """Prompt ablation as a 2x2 of original finding by abstract (the claim tested is always present), token score without reasoning, one panel per set. The bracket is the paired test of the full prompt against no finding. With interaction, each panel also states the interaction of original finding by abstract, (full minus no finding) minus (no abstract minus claim alone). Statistics as in Supplementary Section S12 (analysis.py ablation)."""
    mid = A.ABLATION_MODELS[model]
    sets = (("all", "All claims (n = 274)"), ("human", "Human set (n = 135)"), ("highpower", "High-power set (n = 57)"))
    # (abstract present?, finding present?): condition of the prompt ablation
    cells = {(True, False): "no finding", (True, True): "full", (False, False): "claim alone", (False, True): "no abstract"}
    height = 62 if interaction else 56
    fig = plt.figure(figsize=(DOUBLE_COLUMN_MM / 25.4, height / 25.4))
    for n, (subset, title) in enumerate(sets):
        rows = A.sel(subset)
        x = {c: A.arr([A.token(A.qwen(r, A.ABLATION_CONDITIONS[c], "off", 0, mid)) for r in rows]) for c in cells.values()}
        boot = A.boot_any(x, rows)  # papers resampled on all 274, claims on the other sets
        points = {c: (A.auc(v, rows), *interval(boot[c])) for c, v in x.items()}
        d = boot["full"] - boot["no finding"]
        test = (points["full"][0] - points["no finding"][0], *interval(d), A._boot_p(d))
        for c, want in expected[subset]["points"].items():
            for what, found, e in zip(("AUC", "lower", "upper"), points[c], want):
                check(f"{name} {subset} {c} {what}", found, e)
        for what, found, e in zip(("difference", "lower", "upper", "p"), test, expected[subset]["test"]):
            if e is not None:
                check(f"{name} {subset} full minus no finding {what}", found, e)
        d = d - (boot["no abstract"] - boot["claim alone"])
        inter = ((points["full"][0] - points["no finding"][0]) - (points["no abstract"][0] - points["claim alone"][0]), *interval(d), A._boot_p(d))
        if interaction:
            for what, found, e in zip(("difference", "lower", "upper", "p"), inter, expected[subset]["interaction"]):
                check(f"{name} {subset} interaction {what}", found, e)
        print(f"  {name} {subset}: full minus no finding {test[0]:+.3f} [{test[1]:+.3f}, {test[2]:+.3f}], two-sided p {test[3]:.4f}")

        ax = panel(fig, 12 + n * 57.5, 10, 51, height - 16)
        for abstract, off, colour, style, face, label in ((True, -0.05, TOKEN, "-", TOKEN, "Abstract present"), (False, 0.05, TOKEN_LIGHT, (0, (3, 2)), "white", "Abstract absent")):
            ys = [points[cells[abstract, finding]] for finding in (False, True)]
            xs = [off, 1 + off]
            ax.plot(xs, [a for a, _, _ in ys], color=colour, ls=style, lw=1.0, zorder=2)
            ax.errorbar(xs, [a for a, _, _ in ys], yerr=[[a - lo for a, lo, _ in ys], [hi - a for a, _, hi in ys]], color=colour, elinewidth=0.9, marker="o" if abstract else "s",
                        ms=4.6 if abstract else 4.0, mfc=face, mec=colour, mew=0.9, ls="none", zorder=3, label=label)
        top = max(hi for _, _, hi in points.values())
        bracket(ax, -0.05, 0.95, max(top + 0.03, 0.80), p_text(test[3]), color=TOKEN)
        auc_axis(ax, top=1.125 if interaction else 1.03)
        if interaction:
            ax.text(0.5, 1.118, f"Finding x abstract interaction: {inter[0]:+.3f}\n(95% CI {inter[1]:+.3f} to {inter[2]:+.3f}), {p_text(inter[3])}".replace(" x ", " \u00d7 ").replace("-", "\u2212"),
                    size=5.5, color=INK, ha="center", va="top", linespacing=1.25)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1], ["Absent", "Present"])
        ax.set_xlabel("Original finding")
        ax.set_title(title, size=6.5, weight="bold", pad=3)
        if n:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.legend(loc="upper left", frameon=False, fontsize=6, handlelength=0, handletextpad=0.8, borderpad=0.1, labelspacing=0.5, bbox_to_anchor=(0.04, 0.86 if interaction else 0.99))
        panel_label(fig, 0.5 + n * 57.5 + (0 if n == 0 else 8.5), "abc"[n])
    save(fig, name)


FIG3_EXPECTED = {  # Supplementary Section S12; the interaction's p-values are two-sided
    "all": {"points": {"no finding": (0.564, 0.480, 0.646), "full": (0.687, 0.609, 0.762), "claim alone": (0.639, 0.555, 0.717), "no abstract": (0.662, 0.579, 0.740)},
            "test": (0.123, 0.056, 0.191, None), "interaction": (0.100, 0.023, 0.179, 0.010)},
    "human": {"points": {"no finding": (0.607, 0.511, 0.702), "full": (0.691, 0.603, 0.776), "claim alone": (0.651, 0.557, 0.742), "no abstract": (0.635, 0.541, 0.728)},
              "test": (0.084, 0.009, 0.160, None), "interaction": (0.100, 0.009, 0.190, 0.031)},
    "highpower": {"points": {"no finding": (0.746, 0.608, 0.863), "full": (0.838, 0.723, 0.932), "claim alone": (0.778, 0.652, 0.889), "no abstract": (0.778, 0.649, 0.888)},
                  "test": (0.093, -0.005, 0.196, 0.062), "interaction": (0.093, -0.028, 0.220, 0.135)},
}


def fig3_v2():
    """Figure 3: prompt ablation for Qwen3.8 27B, with the interaction of original finding by abstract stated in each panel."""
    ablation_figure("Qwen3.8 27B", "fig3_v2", FIG3_EXPECTED, interaction=True)


# ---------------------------------------------------------------- Figure 4


def fig4():
    """Reasoning on the high-power set, Qwen3.8 27B. Panel a: the 2x2 of readout (token or stated score) by reasoning (none or complete), with the interaction. Panel b: the share of claims whose answer word is near certain. Statistics as in Supplementary Section S6 (analysis.py thinking_complete and olmo_think)."""
    rows = A.sel("highpower")
    full, _ = A.complete_reasoning()
    off = [A.qwen(r, A.F, "off") for r in rows]
    done = [full[r.row] for r in rows]
    x = {("token", "none"): A.arr([A.token(e) for e in off]), ("stated", "none"): A.arr([e["probability"] / 100 for e in off]),
         ("token", "complete"): A.arr([A.token(e) for e in done]), ("stated", "complete"): A.arr([e["probability"] / 100 for e in done])}
    boot = A.boot_aucs(x, rows)
    points = {k: (A.auc(v, rows), *interval(boot[k])) for k, v in x.items()}
    contrast = lambda d: (d["stated", "complete"] - d["token", "complete"]) - (d["stated", "none"] - d["token", "none"])
    d = contrast(boot)
    inter = (contrast({k: v[0] for k, v in points.items()}), *interval(d), A._boot_p(d))
    near = {k: float(np.mean((x["token", k] > 0.99) | (x["token", k] < 0.01))) for k in ("none", "complete")}
    expected = {("token", "none"): (0.838, 0.723, 0.932), ("stated", "none"): (0.628, 0.498, 0.754), ("token", "complete"): (0.759, 0.623, 0.882), ("stated", "complete"): (0.775, 0.645, 0.890)}
    for k, want in expected.items():
        for what, found, e in zip(("AUC", "lower", "upper"), points[k], want):
            check(f"fig4a {k[0]} score, reasoning {k[1]}, {what}", found, e)
    for what, found, e in zip(("value", "lower", "upper", "p"), inter, (0.226, 0.011, 0.432, 0.038)):
        check(f"fig4a interaction {what}", found, e)
    check("fig4b near certain, no reasoning", near["none"], 0.02, places=2)
    check("fig4b near certain, complete reasoning", near["complete"], 0.96, places=2)

    fig = plt.figure(figsize=(120 / 25.4, 62 / 25.4))

    # panel a
    ax = panel(fig, 12, 10, 64, 46)
    panel_label(fig, 0.5, "a")
    human_bands(ax, 1.93, aggregates=False)
    for score, off_x, colour, style, marker, face, label in (("token", -0.04, TOKEN, "-", "o", TOKEN, "Token score"), ("stated", 0.04, STATED, (0, (3, 2)), "s", "white", "Stated score")):
        ys = [points[score, k] for k in ("none", "complete")]
        xs = [off_x, 1 + off_x]
        ax.plot(xs, [a for a, _, _ in ys], color=colour, ls=style, lw=1.0, zorder=2)
        ax.errorbar(xs, [a for a, _, _ in ys], yerr=[[a - lo for a, lo, _ in ys], [hi - a for a, _, hi in ys]], color=colour, elinewidth=0.9, marker=marker, ms=4.6 if marker == "o" else 4.0,
                    mfc=face, mec=colour, mew=0.9, ls="none", zorder=3, label=label)
    auc_axis(ax, top=1.125)
    ax.text(0.72, 1.118, f"Readout \u00d7 reasoning interaction: {signed(inter[0])}\n(95% CI {signed(inter[1])} to {signed(inter[2])}), {p_text(inter[3])}", size=5.5, color=INK, ha="center", va="top", linespacing=1.25)
    ax.set_xlim(-0.45, 1.95)
    ax.set_xticks([0, 1], ["None", "Complete"])
    ax.set_xlabel("Reasoning", x=0.5 * (0.45 + 1) / 2.4 + 0.5 * 0.45 / 2.4)
    ax.legend(loc="upper right", frameon=False, fontsize=6, handlelength=0, handletextpad=0.8, borderpad=0.1, labelspacing=0.5, bbox_to_anchor=(1.0, 0.86))

    # panel b
    bx = panel(fig, 93, 10, 24, 46)
    panel_label(fig, 81, "b")
    shares = [100 * near["none"], 100 * near["complete"]]
    bx.bar([0, 1], shares, width=0.62, color=TOKEN, lw=0)
    for i, v in enumerate(shares):
        bx.text(i, v + 2, f"{v:.0f}%", size=6, ha="center", va="bottom", color=INK)
    bx.set_ylim(0, 100 * (1.125 - 0.4) / 0.6)  # the same axis height above 100% as panel a has above 1.0
    bx.set_yticks(range(0, 101, 20))
    bx.spines["left"].set_bounds(0, 100)
    bx.set_xlim(-0.6, 1.6)
    bx.set_xticks([0, 1], ["None", "Complete"])
    bx.set_xlabel("Reasoning")
    bx.set_ylabel("Claims with answer word near certain (%)")
    save(fig, "fig4")


# ---------------------------------------------------------------- Figure 5

BASE = "#009E73"  # base models (Okabe-Ito bluish green); chat models take the token score's blue
FIG5_EXPECTED = {  # Supplementary Section S18
    "base Qwen3.5 0.8B": (0.556, 0.399, 0.705), "base Qwen3.5 2B": (0.520, 0.362, 0.677), "base Qwen3.5 4B": (0.661, 0.513, 0.794), "base Olmo 3 7B": (0.676, 0.530, 0.814),
    "base Qwen3.5 9B": (0.752, 0.619, 0.868), "base Qwen3 30B-A3B": (0.761, 0.631, 0.877), "base Olmo 3 32B": (0.779, 0.652, 0.885),
    "chat Qwen3.5 0.8B": (0.477, 0.321, 0.630), "chat Qwen3.5 2B": (0.479, 0.322, 0.636), "chat Qwen3.5 4B": (0.604, 0.450, 0.749), "chat Olmo 3 7B Instruct": (0.593, 0.437, 0.740),
    "chat Qwen3.5 9B": (0.772, 0.639, 0.888), "chat Qwen3.8 27B": (0.838, 0.723, 0.932), "chat Qwen3 30B-A3B Instruct": (0.816, 0.695, 0.912), "chat Olmo 3.1 32B Instruct": (0.760, 0.629, 0.872),
}


def size_figure(name: str) -> None:
    """Panel a: AUC on the high-power set against model size for the 15 models without reasoning (base models by L, chat models by the token score), with the least squares line on log10 parameters. Panel b: Qwen3.8 27B's accuracy by confidence tercile. Statistics as in Supplementary Sections S18 and S13 (analysis.py size and confidence)."""
    rows = A.sel("highpower")
    x, info = {}, {}
    for kind, models in (("base", A.BASE_MODELS), ("chat", A.CHAT_MODELS)):
        for label, m, n in models:
            x[f"{kind} {label}"] = A.olmo_l(rows, A.F, m) if kind == "base" else A.chat_scores(rows, m)[0]
            info[f"{kind} {label}"] = (kind, label, n)
    boot = A.boot_aucs(x, rows)
    points = {k: (A.auc(v, rows), *interval(boot[k])) for k, v in x.items()}
    check("fig5a models", sorted(points), sorted(FIG5_EXPECTED))
    for k, want in FIG5_EXPECTED.items():
        for what, found, e in zip(("AUC", "lower", "upper"), points[k], want):
            check(f"fig5a {k} {what}", found, e)
    keys = list(x)
    size = A.arr([info[k][2] for k in keys])
    aucs = A.arr([points[k][0] for k in keys])
    resampled = np.column_stack([boot[k] for k in keys])
    slope = A._slope(np.log10(size), aucs)
    slopes = A.arr([A._slope(np.log10(size), r) for r in resampled])
    rho = A.spearman(size, aucs)
    rho_p = A._boot_p(A.arr([A.spearman(size, r) for r in resampled]))
    for what, found, e in zip(("slope", "slope lower", "slope upper"), (slope, *interval(slopes)), (0.203, 0.115, 0.288)):
        check(f"fig5a {what}", found, e)
    check("fig5a Spearman rho", rho, 0.88, places=2)
    check("fig5a Spearman p below 0.001", rho_p < 0.001, True)

    crows, cx = A._confidence_rows("highpower", "Qwen token")
    pred, conf = A._loo(cx, crows)
    accuracy, terciles, conf_auc, conf_boot = A.confidence_stats(crows, pred, conf)
    # accuracy among the k most confident claims, in the order that defines the terciles, as a check that the most confident third is the high tercile (claims with equal confidence have no order among themselves, so the third must end a group of equal confidence)
    order = np.argsort(conf, kind="stable")[::-1]
    correct = (pred == A.outcomes(crows))[order]
    kept = np.arange(1, len(correct) + 1)
    among = np.cumsum(correct) / kept
    ends = np.array([k == len(kept) or not np.isclose(conf[order][k], conf[order][k - 1]) for k in kept])
    check("fig5b the most confident third ends a group of equal confidence", bool(ends[len(kept) // 3 - 1]), True)
    check("fig5b accuracy, all claims", float(among[-1]), float(accuracy))
    check("fig5b accuracy, all claims", float(accuracy), 0.649)
    check("fig5b accuracy, most confident third", float(among[len(correct) // 3 - 1]), float(terciles[2]))
    for what, found, e in zip(("low", "middle", "high"), terciles, (0.263, 0.789, 0.895)):
        check(f"fig5b accuracy, {what} tercile", found, e)
    for what, found, e in zip(("AUC", "lower", "upper"), (conf_auc, *interval(conf_boot)), (0.854, 0.731, 0.954)):
        check(f"fig5b confidence {what}", found, e)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_MM / 25.4, 68 / 25.4))

    # panel a
    ax = panel(fig, 12, 11, 112, 53)
    panel_label(fig, 0.5, "a")
    ax.set_xscale("log")
    human_bands(ax, 1.02, aggregates=False, ha="left")
    grid = np.array([0.75, 38.0])
    ax.plot(grid, aucs.mean() + slope * (np.log10(grid) - np.log10(size).mean()), color=INK, lw=0.8, zorder=1)
    for k in keys:
        kind, label, n = info[k]
        a, lo, hi = points[k]
        colour = BASE if kind == "base" else TOKEN
        xpos = n * (0.95 if kind == "base" else 1.05)  # base and chat models of the same size sit side by side
        ax.errorbar(xpos, a, yerr=[[a - lo], [hi - a]], color=colour, elinewidth=0.8, marker="o" if label.startswith("Qwen") else "^", ms=4.4, mfc=colour,
                    mec=colour, mew=0.9, ls="none", zorder=3)
    a38 = points["chat Qwen3.8 27B"]
    ax.annotate("Qwen3.8 27B", xy=(27.78 * 1.05, a38[0]), xytext=(15.5, 0.97), size=5.5, color=INK, ha="right", va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.4, shrinkA=1, shrinkB=3))
    auc_axis(ax, lo=0.3, top=1.0)
    ax.set_ylim(0.3, 0.3 + 0.7 * 1.27)  # the same axis height above 1.0 as panel b
    ax.spines["left"].set_bounds(0.3, 1.0)
    ax.set_xlim(0.7, 42)
    ax.set_xticks([1, 3, 10, 30], ["1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.text(0.02, 0.995, f"{signed(slope)} AUC per tenfold increase in size (95% CI {interval(slopes)[0]:.3f} to {interval(slopes)[1]:.3f})\nSpearman's \u03c1 = {rho:.2f}, {p_text(rho_p)}",
            transform=ax.transAxes, size=5.5, color=INK, ha="left", va="top", linespacing=1.25)
    mark = lambda **kw: Line2D([], [], ls="none", ms=4.4, mew=0.9, **kw)
    handles = [mark(marker="o", mfc=TOKEN, mec=TOKEN), mark(marker="o", mfc=BASE, mec=BASE), mark(marker="o", mfc=HUMAN, mec=HUMAN), mark(marker="^", mfc=HUMAN, mec=HUMAN)]
    labels = ["Chat model", "Base model", "Qwen", "Olmo"]
    ax.legend(handles, labels, loc="lower right", ncols=2, frameon=False, fontsize=5.5, handletextpad=0.2, columnspacing=1.6, borderpad=0.1, labelspacing=0.45, bbox_to_anchor=(0.995, 0.0))

    # panel b
    bx = panel(fig, 144, 11, 36, 53)
    panel_label(fig, 131, "b")
    bx.bar(range(3), terciles, width=0.62, color=TOKEN, lw=0)
    for i, v in enumerate(terciles):
        bx.text(i, v + 0.015, f"{v:.2f}", size=6, ha="center", va="bottom", color=INK)
    bx.set_xlim(-0.6, 2.6)
    bx.set_xticks(range(3), ["Low", "Middle", "High"])
    bx.set_xlabel("Confidence tercile")
    bx.set_ylim(0, 1.27)
    bx.set_yticks(np.arange(0, 1.01, 0.2))
    bx.spines["left"].set_bounds(0, 1)
    bx.set_ylabel("Accuracy of Qwen3.8 27B")
    bx.text(0.5, 0.995, f"AUC of confidence for a correct\nprediction: {conf_auc:.3f}\n(95% CI {interval(conf_boot)[0]:.3f} to {interval(conf_boot)[1]:.3f})", transform=bx.transAxes, size=5.5, color=INK, ha="center", va="top", linespacing=1.25)
    save(fig, name)


def fig5():
    """Model size, and Qwen3.8's confidence as accuracy by tercile."""
    size_figure("fig5")


def figS_ablation30():
    """Supplementary figure: prompt ablation for Qwen3 30B-A3B Instruct, with the interaction stated as in Figure 3."""
    ablation_figure("Qwen3 30B-A3B Instruct", "figS_ablation30", {
        "all": {"points": {"no finding": (0.577, 0.493, 0.659), "full": (0.683, 0.601, 0.758), "claim alone": (0.635, 0.552, 0.713), "no abstract": (0.684, 0.605, 0.759)},
                "test": (0.105, 0.052, 0.162, 0.0004), "interaction": (0.055, -0.009, 0.121, 0.092)},
        "human": {"points": {"no finding": (0.549, 0.450, 0.648), "full": (0.659, 0.567, 0.751), "claim alone": (0.641, 0.546, 0.735), "no abstract": (0.678, 0.587, 0.765)},
                  "test": (0.110, 0.056, 0.168, 0.0002), "interaction": (0.074, -0.002, 0.151, 0.056)},
        "highpower": {"points": {"no finding": (0.679, 0.525, 0.821), "full": (0.816, 0.695, 0.912), "claim alone": (0.824, 0.709, 0.919), "no abstract": (0.782, 0.656, 0.889)},
                      "test": (0.138, 0.053, 0.231, 0.0008), "interaction": (0.179, 0.077, 0.291, 0.0002)},
    }, interaction=True)


FIGURES = {"fig1": fig1, "fig2": fig2, "fig3_v2": fig3_v2, "fig4": fig4, "fig5": fig5, "figS_humanset": figS_humanset, "figS_forecasters30": figS_forecasters30, "figS_ablation30": figS_ablation30}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
