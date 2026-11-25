"""Two checks added after an external review, written to out/<sample>/sensitivity.json. They change no published result.

Usage: python steps/s6_sensitivity.py <KuaiRand-Pure/data> --sample final

1. Videos with no recorded length. The share watched is play time over length, capped at one, so a video whose length
   is recorded as zero counts as fully watched. The check reports how often each reward's top 10% contains such videos,
   then re-ranks each user's videos with a recorded length only, with the same scores, and compares the share-watched
   reward with the same number of those videos drawn at random.
2. Events logged in the same millisecond. Events are ordered by user, time and video id, so an event logged in the same
   millisecond as a target, with a smaller video id, enters the target's history. The check recomputes the within-user
   AUC of section 1 without the test targets that have such an event.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "steps"))
from kr import data as kd, evaluate as ke  # noqa: E402

REWARDS = {"completion": "s3_reward_completion__random_test.npy", "long_view": "s1_transformer__random_test.npy",
           "watch_time": "s3_reward_watch_time__random_test.npy"}
STEP1 = {"lgb_history_summaries": "s1_lgb_history_summaries.npy", "network_no_sequence": "s1_no_sequence__random_test.npy",
         "transformer_no_ranking_loss": "s1_transformer_no_ranking_loss__random_test.npy", "transformer": "s1_transformer__random_test.npy"}


def same_ms_as_previous(user, time):
    """True where the previous row belongs to the same user and has the same timestamp."""
    return np.r_[False, (user[1:] == user[:-1]) & (time[1:] == time[:-1])]


def r(x, k=4):
    return round(float(x), k)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("--sample", default="pilot"); a = ap.parse_args()
    der = ROOT / "derived" / a.sample
    D = kd.prepare(a.data_dir, a.sample)
    te = kd.idx(D, "random", "test"); user = D.user[te]
    O = {k: D.outcomes[k][te] for k in ("long_view", "watch_seconds", "completion", "duration")}
    zero = O["duration"] == 0
    S = {k: np.load(der / f) for k, f in REWARDS.items()}
    out = {"sample": a.sample, "zero_length": {"test_random_exposures": int(len(te)), "with_zero_length": int(zero.sum()),
                                               "share": r(zero.mean())}}
    picks = {}
    for k, s in S.items():
        kk, sums, _ = ke.top_parts(user, s, {"zero": zero.astype(float), "full": (zero & (O["completion"] == 1)).astype(float)})
        picks[k] = {"picks": int(kk.sum()), "with_zero_length": int(sums["zero"].sum()), "share": r(sums["zero"].sum() / kk.sum()),
                    "zero_length_and_fully_watched": int(sums["full"].sum())}
    out["zero_length"]["top10pct"] = picks
    # re-rank among videos with a recorded length only
    keep = ~zero; u2 = user[keep]; O2 = {n: v[keep] for n, v in O.items()}
    rer = {}
    for k, s in S.items():
        kk, sums, base = ke.top_parts(u2, s[keep], O2)
        rer[k] = {n: ke.bootstrap_ratio(sums[n], kk) for n in O2}
        if k == "completion":
            rer["shown_at_random"] = {n: ke.bootstrap_ratio(base[n], kk) for n in O2}
            rer["completion_minus_random__watch_seconds"] = ke.bootstrap_ratio_difference(sums["watch_seconds"], kk, base["watch_seconds"], kk)
            rer["picks"] = int(kk.sum())
    out["zero_length"]["recorded_length_only_top10pct"] = rer

    # events logged in the same millisecond
    flag = same_ms_as_previous(D.user, D.time)[te]
    keep = ~flag; u3 = user[keep]; y3 = O["long_view"][keep]
    P = {k: np.load(der / f)[keep] for k, f in STEP1.items()}
    parts = {k: ke.user_auc_parts(u3, y3, p) for k, p in P.items()}
    out["same_millisecond"] = {
        "test_targets_with_same_ms_previous_event": int(flag.sum()), "share": r(flag.mean()),
        "within_user_auc_without_them": {k: ke.bootstrap_ratio(v[1], v[0]) for k, v in parts.items()},
        "transformer_minus_lgb_history_summaries": ke.bootstrap_ratio_difference(parts["transformer"][1], parts["transformer"][0],
                                                                                 parts["lgb_history_summaries"][1], parts["lgb_history_summaries"][0]),
        "transformer_minus_network_no_sequence": ke.bootstrap_ratio_difference(parts["transformer"][1], parts["transformer"][0],
                                                                               parts["network_no_sequence"][1], parts["network_no_sequence"][0]),
        "transformer_minus_transformer_no_ranking_loss": ke.bootstrap_ratio_difference(parts["transformer"][1], parts["transformer"][0],
                                                                                       parts["transformer_no_ranking_loss"][1], parts["transformer_no_ranking_loss"][0])}
    path = ROOT / "out" / a.sample / "sensitivity.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
