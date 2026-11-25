"""Draw the main figure from out/<sample>/results.json.

Usage: python steps/s_figures.py --sample pilot|final
Colours: navy = the result the study stands behind; grey = a simple baseline; blue = other methods;
orange = a number that misleads.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAVY, BLUE, GREY, ORANGE, INK, MUTED = "#1f3b5c", "#3f78b0", "#9aa3ad", "#e07b39", "#1d232a", "#6b7480"
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#c9ced4",
                     "xtick.color": MUTED, "ytick.color": INK, "savefig.dpi": 200})


def rows(ax, items, xlabel, fmt, xlim=None):
    """items: (label, {'estimate', 'ci95'}, colour, note, note_colour), drawn top to bottom. The row the study stands
    behind (navy) gets a pale band and a bold label; intervals are drawn as rounded pills."""
    n = len(items)
    for i, (label, v, col, note, ncol) in enumerate(items):
        y = n - 1 - i; lo, hi = v["ci95"]; e = v["estimate"]
        if col == NAVY:
            ax.axhspan(y - 0.44, y + 0.44, color="#edf1f6", lw=0, zorder=0)
        ax.plot([lo, hi], [y, y], color=col, lw=7.5, solid_capstyle="round", zorder=2, alpha=0.26)
        ax.scatter([e], [y], s=66, color=col, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(fmt(e), (hi, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.8,
                    fontweight="bold", color=col if col in (NAVY, ORANGE) else INK)
        if note:
            ax.annotate(note, (hi, y), xytext=(8, -11.5), textcoords="offset points", va="center", fontsize=7.2,
                        color=ncol or MUTED, fontweight="bold" if ncol == ORANGE else "normal")
    ax.set_yticks(range(n)); ax.set_yticklabels([it[0] for it in items][::-1])
    for t, it in zip(ax.get_yticklabels(), items[::-1]):
        if it[2] == NAVY:
            t.set_fontweight("bold"); t.set_color(NAVY)
    ax.tick_params(axis="both", length=0); ax.tick_params(axis="y", pad=8); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(xlabel, color=MUTED, labelpad=6); ax.grid(axis="x", color="#ece9e3", lw=0.7); ax.set_axisbelow(True)
    ax.spines["bottom"].set_color("#d6d2ca")
    if xlim:
        ax.set_xlim(*xlim)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", default="final"); a = ap.parse_args()
    R = json.loads((ROOT / "out" / a.sample / "results.json").read_text())
    S4 = json.loads((ROOT / "out" / a.sample / "step4.json").read_text())
    # 2 x 2, landscape. Left column A over C, right column B over D: nine rows each, so every row has the same height.
    fig = plt.figure(figsize=(11.2, 6.4))
    outer = fig.add_gridspec(1, 2, wspace=0.62)
    left = outer[0].subgridspec(2, 1, height_ratios=[5, 4], hspace=0.62)
    right = outer[1].subgridspec(2, 1, height_ratios=[3, 6], hspace=0.62)
    axA, axC = fig.add_subplot(left[0]), fig.add_subplot(left[1])
    axB, axD = fig.add_subplot(right[0]), fig.add_subplot(right[1])
    title = lambda ax, t: ax.set_title(t, loc="left", fontweight="bold", color=INK, pad=9, fontsize=9.6)

    # A: step 1
    M = R["step1"]["models"]; best = max(M, key=lambda k: M[k]["auc"])
    lab = {"lgb_no_history": "Trees, no history", "lgb_history_summaries": "Trees, history summaries",
           "network_no_sequence": "Network, no sequence branch", "transformer_no_ranking_loss": "Transformer, no within-user loss",
           "transformer": "Transformer (rule 1)"}
    col = {"lgb_no_history": GREY, "lgb_history_summaries": GREY, "network_no_sequence": BLUE,
           "transformer_no_ranking_loss": BLUE, "transformer": NAVY}
    items = []
    for k in lab:
        note, nc = f"pooled AUC {M[k]['auc']:.3f}", None
        if k == best and k != "transformer":
            note, nc = f"highest pooled AUC {M[k]['auc']:.3f}", ORANGE
        items.append((lab[k], M[k]["within_user_auc"], col[k], note, nc))
    rows(axA, items, "Within-user AUC, random exposures", lambda v: f"{v:.3f}")
    title(axA, "A   Model: the loss matters, the sequence does not")

    # B: step 2
    P = R["step2"]["policy_value_top10pct"]; off = R["step2"]["usual_offline_check"]
    s, r = off["trained_on_standard_feed"], off["trained_on_random_exposures"]
    items = [("Trained on random exposures", P["trained_on_random_exposures"]["long_view"], NAVY,
              f"standard-feed AUC {r['auc_on_standard_feed']:.3f}", None),
             ("Trained on standard feed", P["trained_on_standard_feed"]["long_view"], BLUE,
              f"standard-feed AUC {s['auc_on_standard_feed']:.3f}; predicts {s['predicted_over_actual_on_random_exposures']:.1f}x", ORANGE),
             ("Shown at random", P["shown_at_random"]["long_view"], GREY, None, None)]
    rows(axB, items, "Long-view rate, top 10% of each user's exposures", lambda v: f"{100 * v:.1f}%")
    axB.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    axB.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    title(axB, "B   Logs: the usual check prefers the worse ranker")

    # C: step 3
    Q = R["step3"]["policy_value_top10pct"]
    note = lambda k: f"completes {100 * Q[k]['completion']['estimate']:.0f}%; {Q[k]['duration']['estimate']:.0f} s videos"
    sens = ROOT / "out" / a.sample / "sensitivity.json"
    zero = json.loads(sens.read_text())["zero_length"]["top10pct"]["completion"]["share"] if sens.exists() else None
    items = [("Reward: watch time", Q["watch_time"]["watch_seconds"], NAVY, note("watch_time"), None),
             ("Reward: long view", Q["long_view"]["watch_seconds"], BLUE, note("long_view"), None),
             ("Reward: share watched", Q["completion"]["watch_seconds"], ORANGE,
              note("completion") if zero is None else f"completes {100 * Q['completion']['completion']['estimate']:.0f}%; {100 * zero:.0f}% with no length", ORANGE),
             ("Shown at random", Q["shown_at_random"]["watch_seconds"], GREY, note("shown_at_random"), None)]
    rows(axC, items, "Seconds watched per exposure, top 10%", lambda v: f"{v:.1f} s")
    title(axC, "C   Reward: share watched loses watch time")

    # D: step 4
    A, V = S4["within_user_auc"], S4["video_level"]
    corr = lambda k: f"correlation {V[k]['weighted_pearson_with_observed_rate']:.2f}"
    items = [("Metadata (rule 4)", A["metadata"], NAVY, corr("metadata"), None),
             ("+ length & caption embedding", A["metadata_frozen_text"], BLUE, corr("metadata_frozen_text"), None),
             ("+ fine-tuned language model", A["metadata_finetuned_text"], BLUE, "kept = metadata model", None),
             ("LLM judge, yes or no", A["zero_shot_judge"], ORANGE, corr("zero_shot_judge"), ORANGE),
             ("LLM judge, rating 0-9", A["zero_shot_judge_rating"], ORANGE, corr("zero_shot_judge_rating"), ORANGE),
             ("Reference: other viewers' rate", A["reference_rate_from_other_viewers"], GREY, "not cold start", None)]
    rows(axD, items, "Within-user AUC, random exposures of new videos", lambda v: f"{v:.3f}")
    axD.axvline(0.5, color=ORANGE, lw=0.9, ls=(0, (3, 3)), zorder=1, alpha=0.7)
    axD.annotate("chance", (0.5, len(items) - 0.45), xytext=(3, 0), textcoords="offset points", fontsize=7.2, color=ORANGE, va="center")
    axD.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    title(axD, "D   New videos: an LLM judge is near chance")
    for ax, pad_left in ((axA, 0.06), (axB, 0.08), (axC, 0.08), (axD, 0.04)):
        lo = min(l.get_xdata()[0] for l in ax.lines if len(l.get_xdata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1])
        hi = max(l.get_xdata()[1] for l in ax.lines if len(l.get_xdata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1])
        span = hi - lo; ax.set_xlim(lo - span * pad_left, hi + span * 0.62)
    axD.set_xlim(0.48, axD.get_xlim()[1])
    out = ROOT / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / f"figure1_{a.sample}.png", bbox_inches="tight")
    print("saved", out / f"figure1_{a.sample}.png")
    if (ROOT / "out" / a.sample / "explorer.json").exists():
        more_figures(a.sample, R, S4)


def pct_axis(ax, axis="y", decimals=0):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=decimals))


def style(ax, title):
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=10)
    ax.grid(axis="y", color="#eef0f2", lw=0.8); ax.set_axisbelow(True)
    ax.spines["bottom"].set_color("#c9ced4"); ax.tick_params(length=0)


def more_figures(sample, R, S4):
    """Figures 2 to 4, from out/<sample>/explorer.json (written by steps/s5_explorer_data.py)."""
    X = json.loads((ROOT / "out" / sample / "explorer.json").read_text())
    sh, C, rnd = X["shares"], X["curves"], X["random"]
    out = ROOT / "figures"
    shade = "#f6d9c4"

    # Figure 2: training logs at every list length, and calibration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.28})
    top, bot = C["transformer"]["long_view"], C["trained_on_standard_feed"]["long_view"]
    a1.fill_between(sh, bot, top, color="#dfe6ee", lw=0)
    for k, lab, col, lw in [("lgb_history_summaries", "Trees, history summaries", GREY, 1.2), ("network_no_sequence", "Network without Transformer", BLUE, 1.2),
                            ("transformer", "Transformer, trained on random exposures", NAVY, 2.6),
                            ("trained_on_standard_feed", "Same network, trained on the standard feed", ORANGE, 2.6)]:
        a1.plot(sh, C[k]["long_view"], color=col, lw=lw, marker="o", ms=3 if lw < 2 else 4, label=lab, zorder=3)
    a1.axhline(rnd["long_view"], color=GREY, ls="--", lw=1); a1.text(0.0105, rnd["long_view"] + 0.004, "shown at random", color=MUTED, fontsize=7.5)
    i = sh.index(0.1)
    a1.annotate(f"{100 * top[i]:.1f}% vs {100 * bot[i]:.1f}%\nat the top 10%", (0.1, (top[i] + bot[i]) / 2), xytext=(0.22, 0.235),
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a1.set_xscale("log"); a1.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]); a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a1.set_ylim(0, 0.26); pct_axis(a1); a1.set_xlabel("Share of each user's ranked list shown (log scale)", color=MUTED)
    a1.set_ylabel("Long-view rate of the videos shown", color=MUTED); a1.legend(frameon=False, fontsize=7.5, loc="lower left")
    style(a1, "A   The training logs move every point of the curve")
    for k, lab, col in [("transformer", "Trained on random exposures", NAVY), ("trained_on_standard_feed", "Trained on the standard feed", ORANGE)]:
        c = X["calibration"][k]; a2.plot(c["predicted"], c["observed"], color=col, lw=2.2, marker="o", ms=4.5, label=lab)
    a2.plot([0, 0.3], [0, 0.3], color=GREY, ls="--", lw=1); a2.text(0.262, 0.222, "predicted = observed", color=MUTED, fontsize=7.5, rotation=40)
    r2 = R["step2"]["usual_offline_check"]["trained_on_standard_feed"]["predicted_over_actual_on_random_exposures"]
    a2.annotate(f"predicts {r2:.1f}x\nthe observed rate", (0.44, 0.137), xytext=(0.33, 0.24), fontsize=8, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    a2.set_xlim(0, 0.58); a2.set_ylim(0, 0.31); pct_axis(a2); pct_axis(a2, "x")
    a2.set_xlabel("Mean predicted long-view rate, tenths of the prediction", color=MUTED); a2.set_ylabel("Observed long-view rate", color=MUTED)
    a2.legend(frameon=False, fontsize=7.5, loc="lower right"); style(a2, "B   The standard-feed model expects its own feed's long views")
    fig.savefig(out / f"figure2_{sample}.png", bbox_inches="tight"); plt.close(fig)

    # Figure 3: what each reward picks
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.25})
    E = X["length_edges"]; labs = ["none" if j == 0 else f"{e:g}–{E[j + 1]:g}" if j + 1 < len(E) else f"{e:g}+" for j, e in enumerate(E)]
    xs = range(len(E)); w = 0.27
    for o, (k, lab, col) in zip((-w, 0, w), [("reward_completion", "Share watched", ORANGE), ("transformer", "Long view", NAVY), ("reward_watch_time", "Seconds watched", BLUE)]):
        a1.bar([x + o for x in xs], X["length_of_picks"][k], width=w, color=col, label=f"Reward: {lab.lower()}", zorder=3)
    a1.step([x - 0.5 for x in xs] + [len(E) - 0.5], X["length_of_picks"]["shown_at_random"] + [X["length_of_picks"]["shown_at_random"][-1]],
            where="post", color=INK, lw=1, ls=":", label="Shown at random", zorder=4)
    a1.set_xticks(list(xs)); a1.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    pct_axis(a1); a1.set_xlabel("Recorded length of the videos in each user's top 10% (seconds; none = recorded as zero)", color=MUTED)
    a1.legend(frameon=False, fontsize=7.5); style(a1, "A   A share-watched reward picks short and zero-length videos")
    for k, lab, col in [("reward_completion", "Share watched", ORANGE), ("transformer", "Long view", NAVY), ("reward_watch_time", "Seconds watched", BLUE)]:
        a2.plot(C[k]["watch_seconds"], C[k]["long_view"], color=col, lw=1.6, marker="o", ms=3, alpha=0.9)
        a2.scatter([C[k]["watch_seconds"][i]], [C[k]["long_view"][i]], s=70, color=col, edgecolor="white", lw=1.2, zorder=4)
        right = k == "reward_watch_time"
        a2.annotate(lab, (C[k]["watch_seconds"][i], C[k]["long_view"][i]), xytext=(4 if right else -8, -16 if right else 8), textcoords="offset points",
                    fontsize=8, color=col, fontweight="bold", ha="left" if right else "right")
    a2.scatter([rnd["watch_seconds"]], [rnd["long_view"]], s=50, color=GREY, zorder=4)
    a2.annotate("at random", (rnd["watch_seconds"], rnd["long_view"]), xytext=(6, -12), textcoords="offset points", fontsize=7.5, color=MUTED)
    a2.set_xlim(0, None); a2.set_ylim(0, None); a2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05)); pct_axis(a2)
    a2.set_xlabel("Seconds watched per exposure", color=MUTED); a2.set_ylabel("Long-view rate", color=MUTED)
    style(a2, "B   Each reward traced from 1% to 100% of the list; dots at 10%")
    fig.savefig(out / f"figure3_{sample}.png", bbox_inches="tight"); plt.close(fig)

    # Figure 4: new videos by score
    cs = X["cold_start"]; S = cs["scatter"]
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.05], "wspace": 0.3})
    import numpy as np
    obs = np.array(S["observed"]); size = 2 + np.sqrt(np.array(S["exposures"])) / 2.2
    for ax, key, lab, col, ck in [(axs[0], "metadata", "Metadata model", NAVY, "metadata"), (axs[1], "judge", "Language-model judge", ORANGE, "zero_shot_judge")]:
        ax.scatter(S[key], np.sqrt(obs), s=size, color=col, alpha=0.22, lw=0)
        dec = cs["by_decile"][ck]; ax.plot([(j + 0.5) / 10 for j in range(10)], np.sqrt(dec), color=col, lw=2.6, marker="o", ms=4.5, mec="white")
        ticks = [0, 0.05, 0.1, 0.2, 0.4]; ax.set_yticks(np.sqrt(ticks)); ax.set_yticklabels([f"{100 * t:.0f}%" for t in ticks])
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["lowest", "median", "highest"]); ax.set_xlabel(f"{lab}: score percentile", color=MUTED)
        c = S4["video_level"][ck]["weighted_pearson_with_observed_rate"]
        ax.text(0.98, 0.97, f"correlation with observed rate {c:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=col, fontweight="bold")
        style(ax, ("A   " if key == "metadata" else "B   ") + lab)
    axs[0].set_ylabel("Observed long-view rate (square-root scale)", color=MUTED)
    for k, lab, col, lw in [("metadata", "Metadata", NAVY, 2.4), ("metadata_frozen_text", "+ caption embedding", BLUE, 1.6),
                            ("zero_shot_judge", "Judge, yes or no", ORANGE, 2.4), ("zero_shot_judge_rating", "Judge, rating", "#f0a878", 1.6)]:
        axs[2].plot(range(1, 11), cs["by_decile"][k], color=col, lw=lw, marker="o", ms=3.5, label=lab)
    axs[2].axhline(cs["overall_rate"], color=GREY, ls="--", lw=1); pct_axis(axs[2]); axs[2].set_xticks(range(1, 11))
    axs[2].set_xlabel("Tenth of the score (1 = lowest)", color=MUTED); axs[2].legend(frameon=False, fontsize=7.5)
    style(axs[2], "C   Long-view rate by tenth of each score")
    fig.savefig(out / f"figure4_{sample}.png", bbox_inches="tight"); plt.close(fig)
    print("saved figures 2 to 4")


if __name__ == "__main__":
    main()
