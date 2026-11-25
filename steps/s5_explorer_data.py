"""Summaries behind the interactive page and the second set of figures, written to out/<sample>/explorer.json.

Usage: python steps/s5_explorer_data.py <KuaiRand-Pure/data> pool_captions.parquet --sample final

Reads the saved test-set scores of steps 1 to 4 (derived/<sample>/) and writes only aggregates: policy values at
several list lengths, calibration by decile, the length of the videos each reward selects, and the observed
long-view rate of new videos by score decile. No row of the data leaves derived/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "steps"))
from kr import data as kd, text as kt  # noqa: E402

SHARES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]
OUT = ["long_view", "watch_seconds", "completion", "duration"]
LENGTH_EDGES = [0, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 1e9]


def top_mask(user, score, share):
    """Rows a policy would show: each user's top `share` of random exposures by score (at least one), ties broken by
    row order. Matches evaluate.top_parts."""
    order = np.lexsort((np.arange(len(user)), -score, user))
    u = user[order]
    start = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]; size = np.diff(np.r_[start, len(u)])
    rank = np.arange(len(u)) - np.repeat(start, size)
    k = np.maximum(1, np.array([int(round(share * n)) for n in size]))
    m = np.zeros(len(user), bool); m[order[rank < np.repeat(k, size)]] = True
    return m


def curves(user, scores, O):
    res = {}
    for name, s in scores.items():
        res[name] = {o: [] for o in OUT}
        for sh in SHARES:
            m = top_mask(user, s, sh)
            for o in OUT:
                res[name][o].append(round(float(O[o][m].mean()), 5))
    return res


def calibration(pred, y, bins=10):
    q = np.quantile(pred, np.linspace(0, 1, bins + 1)); b = np.clip(np.searchsorted(q, pred, side="right") - 1, 0, bins - 1)
    return {"predicted": [round(float(pred[b == i].mean()), 5) for i in range(bins)],
            "observed": [round(float(y[b == i].mean()), 5) for i in range(bins)]}


def length_hist(d, m):
    h = np.histogram(d[m], bins=LENGTH_EDGES)[0]
    return [round(float(x), 5) for x in h / h.sum()]


def cold_start(data_dir, captions, sample, der):
    import s4_captions as s4
    from sklearn.decomposition import PCA
    v, ex = kt.load(data_dir, captions, sample)
    tr, va, te = (np.flatnonzero(v.part.values == p) for p in ("train", "val", "test"))
    X = kt.metadata(v)
    S = {}
    S["metadata"], _, _ = s4.metadata_models(X, v, tr, va)
    E = np.load(der / "s4_embeddings.npy"); Z = PCA(32, random_state=0).fit(E[tr]).transform(E)
    m = s4.fit_lgb(np.column_stack([X, Z]), v, tr, va)
    S["metadata_frozen_text"] = s4.logit(m.predict(np.column_stack([X, Z]), num_iteration=m.best_iteration))
    S["zero_shot_judge"] = np.load(der / "s4_judge.npy")
    S["zero_shot_judge_rating"] = np.load(der / "s4_judge_rating.npy")
    w = v.n_exp.values[te].astype(float); obs = v.rate.values[te]
    out = {"videos": int(len(te)), "exposures": int(w.sum()), "overall_rate": round(float(np.average(obs, weights=w)), 5),
           "by_decile": {}}
    for k, s in S.items():
        s = s[te]; q = np.quantile(s, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, s, side="right") - 1, 0, 9)
        out["by_decile"][k] = [round(float(np.average(obs[b == i], weights=w[b == i])), 5) for i in range(10)]
    # a thinned scatter: every video's score percentile under the metadata model and the judge, and its observed rate
    pct = lambda s: (np.argsort(np.argsort(s)) + 0.5) / len(s)
    out["scatter"] = {"metadata": [round(float(x), 4) for x in pct(S["metadata"][te])],
                      "judge": [round(float(x), 4) for x in pct(S["zero_shot_judge"][te])],
                      "observed": [round(float(x), 4) for x in obs], "exposures": [int(x) for x in w]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("captions")
    ap.add_argument("--sample", default="pilot"); a = ap.parse_args()
    der = ROOT / "derived" / a.sample
    D = kd.prepare(a.data_dir, a.sample)
    te = kd.idx(D, "random", "test"); user = D.user[te]; O = {k: D.outcomes[k][te] for k in OUT}
    load = lambda n: np.load(der / n)
    models = {"lgb_no_history": load("s1_lgb_no_history.npy"), "lgb_history_summaries": load("s1_lgb_history_summaries.npy"),
              "network_no_sequence": load("s1_no_sequence__random_test.npy"),
              "transformer_no_ranking_loss": load("s1_transformer_no_ranking_loss__random_test.npy"),
              "transformer": load("s1_transformer__random_test.npy"),
              "trained_on_standard_feed": load("s2_trained_on_standard__random_test.npy"),
              "reward_completion": load("s3_reward_completion__random_test.npy"),
              "reward_watch_time": load("s3_reward_watch_time__random_test.npy")}
    R = {"sample": a.sample, "shares": SHARES, "length_edges": LENGTH_EDGES[:-1],
         "random": {o: round(float(O[o].mean()), 5) for o in OUT},
         "curves": curves(user, models, O)}
    y = O["long_view"]
    R["calibration"] = {k: calibration(models[k], y) for k in ("transformer", "trained_on_standard_feed")}
    m10 = {k: top_mask(user, models[k], 0.10) for k in ("reward_completion", "transformer", "reward_watch_time")}
    R["length_of_picks"] = {k: length_hist(O["duration"], m) for k, m in m10.items()}
    R["length_of_picks"]["shown_at_random"] = length_hist(O["duration"], np.ones(len(user), bool))
    R["cold_start"] = cold_start(a.data_dir, a.captions, a.sample, der)
    out = ROOT / "out" / a.sample; out.mkdir(parents=True, exist_ok=True)
    (out / "explorer.json").write_text(json.dumps(R))
    i = SHARES.index(0.1)
    print({k: (v["long_view"][i], v["watch_seconds"][i]) for k, v in R["curves"].items()})


if __name__ == "__main__":
    main()
