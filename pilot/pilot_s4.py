"""Pilot of step 4 (pilot videos only): try other fine-tuning settings and another judge prompt, after the first
pilot run (logs/s4_pilot.log) kept no fine-tuned epoch and found no signal in the yes/no judge.

Usage: python pilot/pilot_s4.py <KuaiRand-Pure/data> <pool_captions.parquet>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "steps"))
import s4_captions as s4  # noqa: E402
from kr import text as kt  # noqa: E402

RATING = s4.RATING


def main():
    v, ex = kt.load(sys.argv[1], sys.argv[2], "pilot")
    tr, va, te = (np.flatnonzero(v.part.values == p) for p in ("train", "val", "test"))
    X = kt.metadata(v); texts = kt.prompt_text(v)
    meta, oof, _ = s4.metadata_models(X, v, tr, va)
    row = {x: i for i, x in enumerate(v.video_id.tolist())}
    ex = ex[ex.video_id.map(v.set_index("video_id").part) == "test"].reset_index(drop=True)
    r = ex.video_id.map(row).to_numpy(); u = ex.user_id.to_numpy(); y = ex.long_view.to_numpy().astype(float)
    auc = lambda s: kt.weighted_within_user_auc(u, y, s[r], np.ones(len(y)))
    res = {"metadata": auc(meta)}
    # reference: the video's own long-view rate among the other viewers of its random exposures (not cold start)
    k, n = v.n_long.values[r], v.n_exp.values[r]
    res["reference_rate_from_other_viewers"] = kt.weighted_within_user_auc(u, y, (k - y + 1) / (n - 1 + 12), np.ones(len(y)))
    sub = np.r_[va, te]
    for name, kw in [("top4_lr1e-5", dict(lr=1e-5, top_layers=4)), ("full_lr5e-6", dict(lr=5e-6)),
                     ("top8_lr3e-5", dict(lr=3e-5, top_layers=8))]:
        z, vl = s4.finetune(texts, v, oof, tr, va, 0, **kw)
        res[f"finetune_{name}"] = {"auc": auc(z - oof + meta), "best_val_loss": vl}
        print(name, res[f"finetune_{name}"], flush=True)
    J = np.zeros(len(v)); J[sub] = s4.judge([texts[i] for i in sub], prompt=RATING, scale=True)
    res["judge_rating"] = auc(J)
    J0 = np.load(ROOT / "derived" / "pilot" / "s4_judge.npy"); res["judge_yes_no"] = auc(J0)
    print(json.dumps(res, indent=1))
    (ROOT / "out" / "pilot" / "pilot_s4.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
