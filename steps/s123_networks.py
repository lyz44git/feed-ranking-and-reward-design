"""Train every network of steps 1 to 3 and save its scores on the test users' random exposures.

Usage: python steps/s123_networks.py <KuaiRand-Pure/data> --sample pilot|final

Every network gets the baseline's history summaries and is trained on its label's loss plus a within-user
pairwise loss, unless stated.

Step 1  the Transformer over the last 50 events; the same network without the sequence; the Transformer without
        the pairwise loss. Label: long view. Trained on random exposures.
Step 2  the sequence model trained on standard-feed exposures instead (the logs a team normally has).
Step 3  the sequence model trained on random exposures with other rewards: completion share, watch time.
"""
import argparse
import json
import sys
from pathlib import Path

import sklearn.ensemble  # noqa: F401  (before torch)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kr import data as kd, net as kn  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("--sample", default="pilot")
    a = ap.parse_args()
    out = ROOT / "derived" / a.sample; out.mkdir(parents=True, exist_ok=True)
    D = kd.prepare(a.data_dir, a.sample)
    O = D.outcomes
    te = kd.idx(D, "random", "test")
    rnd = {p: kd.idx(D, "random", p) for p in ("train", "val")}
    std = {p: kd.idx(D, "standard", p) for p in ("train", "val")}
    std_te = kd.idx(D, "standard", "test")
    log_watch = np.log1p(O["watch_seconds"])
    runs = [  # name, label, loss, training target, options
        ("s1_transformer", O["long_view"], "bce", rnd, {}),                                   # the model the policies use
        ("s1_no_sequence", O["long_view"], "bce", rnd, {"use_history": False}),                # same, history as summaries only
        ("s1_transformer_no_ranking_loss", O["long_view"], "bce", rnd, {"rank_weight": 0.0}),
        ("s2_trained_on_standard", O["long_view"], "bce", std, {}),
        ("s3_reward_completion", O["completion"], "bce", rnd, {}),
        ("s3_reward_watch_time", log_watch, "mse", rnd, {}),
    ]
    info = {"rows": {"random_train": len(rnd["train"]), "random_val": len(rnd["val"]), "random_test": len(te),
                     "standard_train": len(std["train"]), "standard_val": len(std["val"]), "standard_test": len(std_te)},
            "users": {p: int(len(np.unique(D.user[D.part == p]))) for p in ("train", "val", "test")},
            "history_length_at_test_exposures": [float(x) for x in np.quantile(D.pos[te], [0.1, 0.5, 0.9])], "runs": {}}
    for name, y, loss, part, opt in runs:
        ev = {"random_test": te}
        if name in ("s1_transformer", "s2_trained_on_standard"):
            ev["standard_test"] = std_te                     # for the usual offline metric on the standard feed
        p, meta = kn.fit_predict(D, part["train"], y, part["val"], ev, loss=loss, summaries=True, name=name, **opt,
                                 log=lambda s: print(s, flush=True))
        for k, v in p.items():
            np.save(out / f"{name}__{k}.npy", v.astype(np.float32))
        info["runs"][name] = meta
    (out / "networks.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
