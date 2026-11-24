"""Score the saved predictions of steps 1 to 3 on the test users and write out/<sample>/results.json.

Usage: python steps/s_summary.py <KuaiRand-Pure/data> --sample pilot|final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kr import data as kd, evaluate as ke  # noqa: E402

REPORTED = ["long_view", "watch_seconds", "completion", "valid_play", "like", "hate", "duration"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("--sample", default="pilot")
    a = ap.parse_args()
    der = ROOT / "derived" / a.sample; out = ROOT / "out" / a.sample; out.mkdir(parents=True, exist_ok=True)
    D = kd.prepare(a.data_dir, a.sample)
    te = kd.idx(D, "random", "test"); st = kd.idx(D, "standard", "test")
    user = D.user[te]; O = {k: D.outcomes[k][te] for k in REPORTED}; y = O["long_view"]
    load = lambda n: np.load(der / n)
    R = {"sample": a.sample, "networks": json.loads((der / "networks.json").read_text()),
         "test": {"users": int(len(np.unique(user))), "random_exposures": int(len(te)),
                  "long_view_rate_random": float(y.mean()), "long_view_rate_standard_feed": float(D.outcomes["long_view"][st].mean()),
                  "watch_seconds_random": float(O["watch_seconds"].mean()),
                  "watch_seconds_standard_feed": float(D.outcomes["watch_seconds"][st].mean())}}

    # ---- step 1: does the order of the history help prediction?
    S1 = {"lgb_no_history": load("s1_lgb_no_history.npy"), "lgb_history_summaries": load("s1_lgb_history_summaries.npy"),
          "network_no_sequence": load("s1_no_sequence__random_test.npy"),
          "transformer_no_ranking_loss": load("s1_transformer_no_ranking_loss__random_test.npy"),
          "transformer": load("s1_transformer__random_test.npy")}
    parts = {k: ke.user_auc_parts(user, y, p) for k, p in S1.items()}
    tp1 = {k: ke.top_parts(user, p, O) for k, p in S1.items()}
    R["step1"] = {"models": {k: {"auc": float(roc_auc_score(y, p)),
                                 "within_user_auc": ke.bootstrap_ratio(parts[k][1], parts[k][0]),
                                 "top10pct_long_view": ke.bootstrap_ratio(tp1[k][1]["long_view"], tp1[k][0]),
                                 "top10pct_watch_seconds": ke.bootstrap_ratio(tp1[k][1]["watch_seconds"], tp1[k][0])}
                             for k, p in S1.items()}}
    for other in ("lgb_history_summaries", "network_no_sequence", "transformer_no_ranking_loss"):
        R["step1"][f"transformer_minus_{other}__within_user_auc"] = ke.bootstrap_ratio_difference(
            parts["transformer"][1], parts["transformer"][0], parts[other][1], parts[other][0])
    d1 = R["step1"]["transformer_minus_lgb_history_summaries__within_user_auc"]
    R["step1"]["rule_met"] = bool(d1["ci95"][0] > 0)

    # ---- step 2: which logs to learn from, and what the usual offline check says
    pol = {"trained_on_random_exposures": S1["transformer"], "trained_on_standard_feed": load("s2_trained_on_standard__random_test.npy")}
    tp = {k: ke.top_parts(user, p, O) for k, p in pol.items()}
    val = {k: {n: ke.bootstrap_ratio(v[1][n], v[0]) for n in REPORTED} for k, v in tp.items()}
    k0, _, base = next(iter(tp.values()))
    val["shown_at_random"] = {n: ke.bootstrap_ratio(base[n], k0) for n in REPORTED}
    a_, b_ = tp["trained_on_random_exposures"], tp["trained_on_standard_feed"]
    ys = D.outcomes["long_view"][st]
    offline = {}
    for k, f in (("trained_on_random_exposures", "s1_transformer"), ("trained_on_standard_feed", "s2_trained_on_standard")):
        ps, pr = load(f"{f}__standard_test.npy"), load(f"{f}__random_test.npy")
        offline[k] = {"auc_on_standard_feed": float(roc_auc_score(ys, ps)), "auc_on_random_exposures": float(roc_auc_score(y, pr)),
                      "within_user_auc_on_random_exposures": ke.within_user_auc(user, y, pr),
                      "mean_prediction_on_random_exposures": float(pr.mean()),
                      "predicted_over_actual_on_random_exposures": float(pr.mean() / y.mean())}
    R["step2"] = {"policy_value_top10pct": val, "usual_offline_check": offline,
                  "difference_long_view": ke.bootstrap_ratio_difference(a_[1]["long_view"], a_[0], b_[1]["long_view"], b_[0]),
                  "difference_watch_seconds": ke.bootstrap_ratio_difference(a_[1]["watch_seconds"], a_[0], b_[1]["watch_seconds"], b_[0])}
    R["step2"]["rule_met"] = bool(R["step2"]["difference_long_view"]["ci95"][0] > 0)

    # ---- step 3: what each reward makes the policy show
    rew = {"completion": load("s3_reward_completion__random_test.npy"), "long_view": S1["transformer"],
           "watch_time": load("s3_reward_watch_time__random_test.npy")}
    rp = {k: ke.top_parts(user, p, O) for k, p in rew.items()}
    R["step3"] = {"policy_value_top10pct": {k: {n: ke.bootstrap_ratio(v[1][n], v[0]) for n in REPORTED} for k, v in rp.items()}}
    R["step3"]["policy_value_top10pct"]["shown_at_random"] = val["shown_at_random"]
    w, c = rp["watch_time"], rp["completion"]
    for n in ("watch_seconds", "long_view", "completion", "duration"):
        R["step3"][f"watch_time_minus_completion__{n}"] = ke.bootstrap_ratio_difference(w[1][n], w[0], c[1][n], c[0])
    R["step3"]["completion_minus_random__watch_seconds"] = ke.bootstrap_ratio_difference(c[1]["watch_seconds"], c[0], base["watch_seconds"], k0)
    R["step3"]["rule_met"] = bool(R["step3"]["watch_time_minus_completion__watch_seconds"]["ci95"][0] > 0)
    (out / "results.json").write_text(json.dumps(R, indent=1))

    r = lambda x: round(x, 4)
    print("step 1", {k: (r(v["auc"]), r(v["within_user_auc"]["estimate"]), r(v["top10pct_long_view"]["estimate"])) for k, v in R["step1"]["models"].items()})
    print("step 1 diffs", {k: v for k, v in R["step1"].items() if k.startswith("transformer_minus")})
    print("step 2 value", {k: {n: r(v[n]["estimate"]) for n in ("long_view", "watch_seconds")} for k, v in val.items()})
    print("step 2 diff", R["step2"]["difference_long_view"], "offline", offline)
    print("step 3", {k: {n: r(v[n]["estimate"]) for n in REPORTED} for k, v in R["step3"]["policy_value_top10pct"].items()})
    print("step 3 diff watch", R["step3"]["watch_time_minus_completion__watch_seconds"], "completion vs random", R["step3"]["completion_minus_random__watch_seconds"])


if __name__ == "__main__":
    main()
