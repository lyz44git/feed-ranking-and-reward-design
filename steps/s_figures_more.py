"""Figures 2 to 4, drawn from out/<sample>/explorer.json, results.json and step4.json.

Called by steps/s_figures.py. The look follows the interactive page (docs/): navy, blue and grey, orange kept for the
number that misleads, thin warm grids, direct labels instead of legends, and the key numbers written on the chart.
Every number is read from the result files.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID = "#1d232b", "#4b5563", "#7b8490", "#e9e6df"
NAVY, BLUE, LIGHT, PALE, GREY, ORANGE, OSOFT = "#1f3b5c", "#3f78b0", "#a9c4de", "#dfe6ee", "#9aa3ad", "#e07b39", "#f6d9c4"
HALO = [patheffects.withStroke(linewidth=3, foreground="white")]
STYLE = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": "#b9bec5", "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white", "lines.solid_capstyle": "round", "savefig.dpi": 300}


def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", color=INK, pad=10)


def dot(ax, x, y, color, s=46, z=5):
    ax.scatter([x], [y], s=s, color=color, edgecolor="white", linewidth=1.2, zorder=z)


def say(ax, x, y, s, color=INK, **kw):
    kw.setdefault("fontsize", 8); kw.setdefault("fontweight", "bold")
    return ax.annotate(s, (x, y), color=color, path_effects=HALO, **kw)


def pct(ax, axis="y", d=0):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(mt.PercentFormatter(1.0, decimals=d))


def share_axis(ax):
    ax.set_xscale("log"); ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")); ax.xaxis.set_minor_locator(mt.NullLocator())


def figure2(X, R, out):
    """Training logs at every list length; the offline check's reversal; calibration."""
    sh, C, rnd = X["shares"], X["curves"], X["random"]; i = sh.index(0.1)
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 3.2), gridspec_kw={"width_ratios": [1.3, 0.8, 1.0], "wspace": 0.45})

    a = ax[0]
    five = np.array([C[k]["long_view"] for k in ("lgb_no_history", "lgb_history_summaries", "network_no_sequence",
                                                 "transformer_no_ranking_loss", "transformer")])
    a.fill_between(sh, five.min(0), five.max(0), color=LIGHT, alpha=0.55, lw=0, zorder=2)
    top, bot = C["transformer"]["long_view"], C["trained_on_standard_feed"]["long_view"]
    a.plot(sh, top, color=NAVY, lw=2.4, zorder=4); a.plot(sh, bot, color=ORANGE, lw=2.4, zorder=4)
    a.axhline(rnd["long_view"], color=GREY, lw=1, ls=(0, (4, 3)), zorder=3)
    dot(a, 0.1, top[i], NAVY); dot(a, 0.1, bot[i], ORANGE)
    d = R["step2"]["difference_long_view"]["estimate"]
    a.annotate("", (0.1, bot[i] + 0.004), xytext=(0.1, top[i] - 0.004), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    say(a, 0.1, (top[i] + bot[i]) / 2, f"+{100 * d:.1f} pp", INK, xytext=(-6, -3), textcoords="offset points", ha="right", fontsize=8.5)
    say(a, 0.0105, top[0] + 0.012, "trained on random exposures", NAVY, fontsize=7.6)
    say(a, 0.0105, five.min(0)[0] - 0.024, "five models, same logs", "#5d7fa6", fontsize=7, fontweight="normal")
    say(a, 0.0105, bot[0] + 0.012, "trained on the standard feed", ORANGE, fontsize=7.6)
    say(a, 0.0105, rnd["long_view"] - 0.019, "shown at random", MUTED, fontsize=7, fontweight="normal")
    say(a, 0.1, top[i], f"{100 * top[i]:.1f}%", NAVY, xytext=(6, 5), textcoords="offset points")
    say(a, 0.1, bot[i], f"{100 * bot[i]:.1f}%", ORANGE, xytext=(4, -15), textcoords="offset points")
    share_axis(a); a.set_ylim(0, 0.265); pct(a)
    a.set_xlabel("Share of each user's ranked list shown (log scale)"); a.set_ylabel("Long-view rate of the videos shown")
    panel(a, "A", "Training logs move every point")

    b = ax[1]
    off = R["step2"]["usual_offline_check"]; pv = R["step2"]["policy_value_top10pct"]
    cols = [("AUC,\nstandard feed", lambda k: off[k]["auc_on_standard_feed"], lambda v: f"{v:.3f}"),
            ("Long views,\ntop 10%", lambda k: pv[k]["long_view"]["estimate"], lambda v: f"{100 * v:.1f}%")]
    for k, col in (("trained_on_random_exposures", NAVY), ("trained_on_standard_feed", ORANGE)):
        ys = []
        for j, (_, f, fm) in enumerate(cols):
            other = "trained_on_standard_feed" if k == "trained_on_random_exposures" else "trained_on_random_exposures"
            ys.append(1.0 if f(k) > f(other) else 0.0)
        b.plot([0, 1], ys, color=col, lw=2.6, zorder=3)
        for j, (_, f, fm) in enumerate(cols):
            dot(b, j, ys[j], col, s=60)
            say(b, j, ys[j], fm(f(k)), col, xytext=(-8 if j == 0 else 8, 0), textcoords="offset points",
                ha="right" if j == 0 else "left", va="center", fontsize=8.5)
    b.set_xlim(-0.75, 1.75); b.set_ylim(-0.35, 1.35); b.set_yticks([0, 1]); b.set_yticklabels(["2nd", "1st"])
    b.set_xticks([0, 1]); b.set_xticklabels([c[0] for c in cols]); b.grid(False); b.spines["bottom"].set_visible(False)
    b.text(0.5, -0.3, "the order reverses", color=MUTED, fontsize=7.2, ha="center")
    panel(b, "B", "The usual check reverses")

    c = ax[2]
    for k, col, lab in (("transformer", NAVY, "random exposures"), ("trained_on_standard_feed", ORANGE, "standard feed")):
        cal = X["calibration"][k]; c.plot(cal["predicted"], cal["observed"], color=col, lw=2.2, zorder=3)
        c.scatter(cal["predicted"], cal["observed"], s=16, color=col, edgecolor="white", lw=0.8, zorder=4)
    c.plot([0, 0.31], [0, 0.31], color=GREY, lw=1, ls=(0, (4, 3)), zorder=2)
    r2 = off["trained_on_standard_feed"]["predicted_over_actual_on_random_exposures"]
    cs = X["calibration"]["trained_on_standard_feed"]
    say(c, cs["predicted"][-1], cs["observed"][-1], f"standard feed:\npredicts {r2:.1f}x", ORANGE, xytext=(-4, 12), textcoords="offset points", ha="right", fontsize=7.6)
    ct = X["calibration"]["transformer"]
    say(c, 0.03, 0.97, "random exposures:\non the diagonal", NAVY, xycoords="axes fraction", va="top", fontsize=7.6)
    c.set_xlim(0, 0.58); c.set_ylim(0, 0.31); pct(c); pct(c, "x")
    c.set_xlabel("Predicted long-view rate (tenths)"); c.set_ylabel("Observed")
    panel(c, "C", "Calibration")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figure3(X, R, out):
    """Length of the picks under each reward; what the share-watched reward gains and loses; the trade-off."""
    E = X["length_edges"] + [600]; L = X["length_of_picks"]; Q = R["step3"]["policy_value_top10pct"]
    sh, C, rnd = X["shares"], X["curves"], X["random"]; i = sh.index(0.1)
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 3.3), gridspec_kw={"width_ratios": [1.15, 0.95, 1.0], "wspace": 0.42})

    a = ax[0]
    edges = np.array([2.0] + E[1:], float)                  # the first bin (0 to 5 s) drawn from 2 s on the log axis
    rows_ = [("shown_at_random", "at random", GREY, "shown_at_random"), ("reward_watch_time", "seconds watched", BLUE, "watch_time"),
             ("transformer", "long view", NAVY, "long_view"), ("reward_completion", "share watched", ORANGE, "completion")]
    for r, (k, lab, col, q) in enumerate(rows_):
        dens = np.array(L[k]) / np.diff(np.log10(edges)); dens = dens / dens.max() * 0.82
        xs = np.repeat(edges, 2)[1:-1]; ys = r + np.repeat(dens, 2)
        a.fill_between(xs, r, ys, color=col, alpha=0.85 if k != "shown_at_random" else 0.45, lw=0, zorder=3 + r)
        a.plot(xs, ys, color="white", lw=0.8, zorder=3 + r)
        mean = Q[q]["duration"]["estimate"]
        a.text(1.9, r + 0.32, lab, color=col if k != "shown_at_random" else MUTED, fontsize=7.6, fontweight="bold", ha="right", va="center")
        a.text(1.9, r + 0.08, f"mean {mean:.0f} s", color=INK2, fontsize=7, ha="right", va="center")
    a.set_xscale("log"); a.set_xlim(2, 600); a.set_ylim(-0.1, 4.0); a.set_yticks([]); a.grid(False)
    a.set_xticks([5, 10, 30, 60, 180, 600]); a.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}")); a.xaxis.set_minor_locator(mt.NullLocator())
    a.set_xlabel("Video length in the top 10% (seconds, log scale)")
    panel(a, "A", "What each reward puts on top")

    b = ax[1]
    mets = [("Share of the\nvideo watched", "completion", lambda v: f"{100 * v:.0f}%"), ("Long-view\nrate", "long_view", lambda v: f"{100 * v:.1f}%"),
            ("Seconds\nwatched", "watch_seconds", lambda v: f"{v:.1f} s")]
    base = Q["shown_at_random"]
    for j, (lab, n, fm) in enumerate(mets):
        y = len(mets) - 1 - j
        for off_, q, col in ((0.17, "completion", ORANGE), (-0.17, "watch_time", BLUE)):
            ch = Q[q][n]["estimate"] / base[n]["estimate"] - 1
            b.barh(y + off_, ch, height=0.3, color=col, zorder=3)
            say(b, max(ch, 0), y + off_, f"{'+' if ch >= 0 else '−'}{abs(100 * ch):.0f}%", col, xytext=(4, 0),
                textcoords="offset points", ha="left", va="center", fontsize=7.6)
    b.axvline(0, color=INK2, lw=0.8); b.set_yticks([])
    for j, (lab, _, _) in enumerate(mets):
        b.text(-0.62, len(mets) - 1 - j, lab, color=INK2, fontsize=7.4, ha="right", va="center")
    b.set_xlim(-0.6, 3.5); b.set_ylim(-1.05, 2.5)
    b.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{100 * v:+.0f}%")); b.set_xticks([0, 1, 2, 3])
    b.grid(axis="x", color=GRID, lw=0.7); b.grid(axis="y", visible=False)
    b.set_xlabel("Change from videos shown at random")
    b.text(3.45, -0.62, "share-watched reward", color=ORANGE, fontsize=7.2, fontweight="bold", ha="right")
    b.text(3.45, -0.92, "seconds-watched reward", color=BLUE, fontsize=7.2, fontweight="bold", ha="right")
    panel(b, "B", "Share watched is gamed")

    c = ax[2]
    for k, lab, col, dy in (("reward_completion", "share watched", ORANGE, 10), ("transformer", "long view", NAVY, 8), ("reward_watch_time", "seconds watched", BLUE, None)):
        c.plot(C[k]["watch_seconds"], C[k]["long_view"], color=col, lw=1.8, alpha=0.9, zorder=3)
        c.scatter(C[k]["watch_seconds"], C[k]["long_view"], s=9, color=col, zorder=4, lw=0)
        dot(c, C[k]["watch_seconds"][i], C[k]["long_view"][i], col, s=58)
        say(c, C[k]["watch_seconds"][i], C[k]["long_view"][i], lab, col, xytext=(-6, dy) if dy is not None else (8, -12),
            textcoords="offset points", ha="right" if dy is not None else "left", fontsize=7.6)
    dot(c, rnd["watch_seconds"], rnd["long_view"], GREY, s=50)
    say(c, rnd["watch_seconds"], rnd["long_view"], "at random", MUTED, xytext=(6, -12), textcoords="offset points", fontsize=7, fontweight="normal")
    c.set_xlim(0, 17); c.set_ylim(0, 0.24); pct(c); c.yaxis.set_major_locator(mt.MultipleLocator(0.05))
    c.set_xlabel("Seconds watched per exposure"); c.set_ylabel("Long-view rate")
    say(c, 16.6, 0.012, "paths: top 1% to 100%; dots: top 10%", MUTED, ha="right", fontsize=6.8, fontweight="normal")
    panel(c, "C", "No reward wins both")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figure4(X, S4, out):
    """New videos sorted by the metadata model and by the language-model judge."""
    cs = X["cold_start"]; S = cs["scatter"]; obs = np.sqrt(np.array(S["observed"])); size = 1.5 + np.sqrt(np.array(S["exposures"])) / 2.6
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.3})
    ticks = [0, 0.05, 0.1, 0.2, 0.4]
    for a, key, ck, lab, col, letter in ((ax[0], "metadata", "metadata", "Metadata model", NAVY, "A"),
                                         (ax[1], "judge", "zero_shot_judge", "Language-model judge", ORANGE, "B")):
        a.scatter(S[key], obs, s=size, color=col, alpha=0.16, lw=0, zorder=2)
        dec = np.sqrt(cs["by_decile"][ck]); xs = [(j + 0.5) / 10 for j in range(10)]
        a.plot(xs, dec, color="white", lw=4.5, zorder=3); a.plot(xs, dec, color=col, lw=2.4, zorder=4)
        a.scatter(xs, dec, s=18, color=col, edgecolor="white", lw=0.9, zorder=5)
        auc = S4["within_user_auc"][ck]["estimate"]; r = S4["video_level"][ck]["weighted_pearson_with_observed_rate"]
        say(a, 0.03, 0.95, f"within-user AUC {auc:.3f}\ncorrelation {r:.2f}", col, xycoords="axes fraction", va="top", fontsize=7.8)
        a.set_yticks(np.sqrt(ticks)); a.set_yticklabels([f"{100 * t:.0f}%" for t in ticks]); a.set_ylim(-0.02, np.sqrt(0.62))
        a.set_xticks([0, 0.5, 1]); a.set_xticklabels(["lowest", "median", "highest"]); a.set_xlabel(f"{lab}: score percentile")
        panel(a, letter, lab)
    ax[0].set_ylabel("Observed long-view rate (square-root scale)")
    c = ax[2]
    for k, col, lw, lab, dy in (("zero_shot_judge_rating", "#f0b48c", 1.3, "judge, rating", 0), ("metadata_frozen_text", LIGHT, 1.3, "", 0),
                                ("zero_shot_judge", ORANGE, 2.4, "judge, yes or no", -9), ("metadata", NAVY, 2.4, "metadata", 0)):
        v = cs["by_decile"][k]; c.plot(range(1, 11), v, color=col, lw=lw, zorder=4 if lw > 2 else 3)
        if lw > 2:
            c.scatter(range(1, 11), v, s=16, color=col, edgecolor="white", lw=0.8, zorder=5)
        if lab:
            say(c, 10, v[-1], lab, col, xytext=(6, dy), textcoords="offset points", va="center", fontsize=7.4)
    c.axhline(cs["overall_rate"], color=GREY, lw=1, ls=(0, (4, 3)), zorder=2)
    say(c, 1, cs["overall_rate"], "all test videos", MUTED, xytext=(0, 5), textcoords="offset points", fontsize=6.8, fontweight="normal")
    pct(c); c.set_xticks(range(1, 11)); c.set_xlim(0.6, 12.6); c.set_xlabel("Tenth of the score (1 = lowest)")
    c.spines["bottom"].set_bounds(1, 10)
    panel(c, "C", "Only metadata sorts the videos")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def draw(sample, R, S4):
    X = json.loads((ROOT / "out" / sample / "explorer.json").read_text())
    with plt.rc_context(STYLE):
        out = ROOT / "figures"
        figure2(X, R, out / f"figure2_{sample}.png")
        figure3(X, R, out / f"figure3_{sample}.png")
        figure4(X, S4, out / f"figure4_{sample}.png")
    print("saved figures 2 to 4")
