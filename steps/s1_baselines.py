"""Step 1, baselines without the order of the history: gradient-boosted trees with and without history summaries.

Usage: python steps/s1_baselines.py <KuaiRand-Pure/data> --sample pilot|final
Run in its own process, before the PyTorch scripts.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kr import data as kd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("--sample", default="pilot")
    a = ap.parse_args()
    out = ROOT / "derived" / a.sample; out.mkdir(parents=True, exist_ok=True)
    D = kd.prepare(a.data_dir, a.sample)
    y = D.outcomes["long_view"]
    tr, va, te = (kd.idx(D, "random", p) for p in ("train", "val", "test"))
    rate = kd.item_rate(D, tr, y)
    base = kd.base_features(D, rate); agg = kd.history_aggregates(D)
    for name, X in (("lgb_no_history", base), ("lgb_history_summaries", np.column_stack([base, agg]))):
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.04, num_leaves=47, min_child_samples=50, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=kd.SEED, n_jobs=8, verbose=-1)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        np.save(out / f"s1_{name}.npy", m.predict_proba(X[te])[:, 1].astype(np.float32))
        print(name, "trees", m.best_iteration_, flush=True)


if __name__ == "__main__":
    main()
