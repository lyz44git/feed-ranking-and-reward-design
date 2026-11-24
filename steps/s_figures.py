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
    """items: (label, {'estimate', 'ci95'}, colour, note, note_colour), drawn top to bottom."""
    n = len(items)
    for i, (label, v, col, note, ncol) in enumerate(items):
        y = n - 1 - i; lo, hi = v["ci95"]; e = v["estimate"]
        ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=2, alpha=0.55)
        ax.scatter([e], [y], s=48, color=col, zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(fmt(e), (hi, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5,
                    fontweight="bold", color=INK)
        if note:
            ax.annotate(note, (hi, y), xytext=(6, -11), textcoords="offset points", va="center", fontsize=7.3,
                        color=ncol or MUTED, fontweight="bold" if ncol == ORANGE else "normal")
    ax.set_yticks(range(n)); ax.set_yticklabels([it[0] for it in items][::-1])
    ax.tick_params(axis="y", length=0); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(xlabel, color=MUTED); ax.grid(axis="x", color="#eef0f2", lw=0.8); ax.set_axisbelow(True)
    if xlim:
        ax.set_xlim(*xlim)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", default="final"); a = ap.parse_args()
    R = json.loads((ROOT / "out" / a.sample / "results.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), gridspec_kw={"height_ratios": [5, 3, 4], "hspace": 0.75})

    # A: step 1
    M = R["step1"]["models"]; best = max(M, key=lambda k: M[k]["auc"])
    lab = {"lgb_no_history": "Trees, no history", "lgb_history_summaries": "Trees, history summaries",
           "network_no_sequence": "Network without the Transformer", "transformer_no_ranking_loss": "Transformer, no within-user loss",
           "transformer": "Transformer (rule 1)"}
    col = {"lgb_no_history": GREY, "lgb_history_summaries": GREY, "network_no_sequence": BLUE,
           "transformer_no_ranking_loss": BLUE, "transformer": NAVY}
    items = []
    for k in lab:
        note, nc = f"overall AUC {M[k]['auc']:.3f}", None
        if k == best and k != "transformer":
            note, nc = f"highest overall AUC ({M[k]['auc']:.3f})", ORANGE
        items.append((lab[k], M[k]["within_user_auc"], col[k], note, nc))
    rows(axes[0], items, "Within-user AUC on random exposures (pairs of the same user)", lambda v: f"{v:.3f}")
    axes[0].set_title("A   Within a user, the training loss matters more than reading the history in order",
                      loc="left", fontweight="bold", color=INK, pad=10)

    # B: step 2
    P = R["step2"]["policy_value_top10pct"]; off = R["step2"]["usual_offline_check"]
    s, r = off["trained_on_standard_feed"], off["trained_on_random_exposures"]
    items = [("Trained on random exposures", P["trained_on_random_exposures"]["long_view"], NAVY,
              f"AUC on standard feed {r['auc_on_standard_feed']:.2f}", None),
             ("Trained on standard feed", P["trained_on_standard_feed"]["long_view"], BLUE,
              f"AUC on standard feed {s['auc_on_standard_feed']:.2f}; predicts {s['predicted_over_actual_on_random_exposures']:.1f}x actual",
              ORANGE),
             ("Shown at random", P["shown_at_random"]["long_view"], GREY, None, None)]
    rows(axes[1], items, "Long-view rate of the top 10% of each user's random exposures", lambda v: f"{100 * v:.1f}%")
    axes[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].set_title("B   The usual offline check prefers the model that recommends worse",
                      loc="left", fontweight="bold", color=INK, pad=10)

    # C: step 3
    Q = R["step3"]["policy_value_top10pct"]
    note = lambda k: f"completes {100 * Q[k]['completion']['estimate']:.0f}%; videos of {Q[k]['duration']['estimate']:.0f} s"
    items = [("Reward: watch time", Q["watch_time"]["watch_seconds"], NAVY, note("watch_time"), None),
             ("Reward: long view", Q["long_view"]["watch_seconds"], BLUE, note("long_view"), None),
             ("Reward: completion share", Q["completion"]["watch_seconds"], ORANGE, note("completion"), ORANGE),
             ("Shown at random", Q["shown_at_random"]["watch_seconds"], GREY, note("shown_at_random"), None)]
    rows(axes[2], items, "Seconds watched per exposure, top 10% of each user's random exposures", lambda v: f"{v:.1f} s")
    axes[2].set_title("C   A completion reward picks very short videos and loses watch time",
                      loc="left", fontweight="bold", color=INK, pad=10)
    for ax in axes:
        lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi + (hi - lo) * 0.55)
    out = ROOT / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / f"figure1_{a.sample}.png", bbox_inches="tight")
    print("saved", out / f"figure1_{a.sample}.png")


if __name__ == "__main__":
    main()
