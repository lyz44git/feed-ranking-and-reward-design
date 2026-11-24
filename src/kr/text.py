"""Step 4: score a video that has no viewing record from its caption, and check the score against random exposures.

Unit: the video. Samples are of videos, not users: `pilot` is video_id % 5 == 0, `final` the rest; within a sample,
videos are split 50 / 10 / 40 into train, validation and test. The label of a video is its long-view rate over its
main-feed random exposures (all users), which the recommender did not choose. Scores are judged on the test
videos' random exposures: within-user AUC, with intervals from resampling test videos.
"""
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .data import SEED, MAIN_TAB

UPLOAD_TYPES = ["ShortImport", "LongImport", "Kmovie", "FollowShoot", "LongCamera", "ShortCamera", "Web", "LongPicture",
                "PictureSet", "PhotoCopy", "ShareFromOtherApp", "LipsSync"]


def video_filter(sample):
    if sample not in ("pilot", "final"):
        raise ValueError(sample)
    return "video_id % 5 = 0" if sample == "pilot" else "video_id % 5 <> 0"


def split_videos(videos, seed=SEED + 4):
    """Deterministic 50 / 10 / 40 split of sorted video ids."""
    videos = np.sort(np.asarray(videos)); perm = np.random.default_rng(seed).permutation(len(videos))
    part = np.empty(len(videos), "U5"); a, b = int(0.5 * len(videos)), int(0.6 * len(videos))
    part[perm[:a]] = "train"; part[perm[a:b]] = "val"; part[perm[b:]] = "test"
    return dict(zip(videos.tolist(), part.tolist()))


def clean_caption(s):
    """Drop account mentions such as '@name(O123)' and repeated spaces; keep words and hashtags."""
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = re.sub(r"@\S*?\(O\d+\)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load(data_dir, captions, sample):
    """Videos of the sample with metadata, caption, part and label counts; and their main-feed random exposures."""
    d = Path(data_dir); c = duckdb.connect(); w = video_filter(sample)
    ex = c.sql(f"""select user_id, video_id, long_view::int as long_view, play_time_ms / 1000.0 as watch_seconds
                   from read_csv_auto('{d / 'log_random_4_22_to_5_08_pure.csv'}')
                   where tab = {MAIN_TAB} and {w} order by user_id, video_id""").df()
    v = c.sql(f"select * from read_csv_auto('{d / 'video_features_basic_pure.csv'}') where {w}").df()
    cap = pd.read_parquet(captions)[["final_video_id", "caption"]].rename(columns={"final_video_id": "video_id"})
    v = v.merge(cap, on="video_id", how="left")
    v["text"] = v.caption.map(clean_caption)
    stats = ex.groupby("video_id").long_view.agg(["sum", "count"]).rename(columns={"sum": "n_long", "count": "n_exp"})
    v = v.merge(stats, on="video_id", how="inner")                    # videos with any main-feed random exposure
    parts = split_videos(v.video_id.to_numpy()); v["part"] = v.video_id.map(parts)
    v["rate"] = v.n_long / v.n_exp
    ex = ex[ex.video_id.isin(v.video_id)].reset_index(drop=True)
    return v.sort_values("video_id").reset_index(drop=True), ex


def metadata(v):
    """What a platform knows about a new video before anyone watches it, apart from its text."""
    dur = v.video_duration.fillna(v.video_duration.median()) / 1000.0
    up = np.column_stack([(v.upload_type == t).to_numpy() for t in UPLOAD_TYPES]).astype(np.float32)
    tag = v.tag.astype(str).str.split(",").str[0]
    tag = pd.to_numeric(tag, errors="coerce").fillna(-1).to_numpy()
    age = (pd.Timestamp("2022-04-22") - pd.to_datetime(v.upload_dt, errors="coerce")).dt.days.fillna(0).to_numpy()
    X = np.column_stack([dur, np.log1p(dur), tag, (v.video_type == "AD").to_numpy(), age, v.music_type.fillna(-1),
                         v.server_height.fillna(0) / np.maximum(v.server_width.fillna(1), 1), up,
                         v.text.str.len().to_numpy()])
    return X.astype(np.float32)


def prompt_text(v):
    """The text the language models read: length in seconds, then the caption."""
    secs = (v.video_duration.fillna(0) / 1000.0).round().astype(int)
    return [f"时长{s}秒。{t if t else '（无标题）'}" for s, t in zip(secs, v.text)]


# --- evaluation: within-user AUC with weights, so test videos can be resampled -------------------------------------

def weighted_within_user_auc(user, y, score, w):
    """Pooled over users: weighted share of (long view, not) pairs of the same user that the score orders
    correctly, ties counted half. Rows must be sorted by user."""
    o = np.lexsort((score, user)); u, yy, s, ww = user[o], y[o], score[o], w[o]
    key_new = np.r_[True, (u[1:] != u[:-1]) | (s[1:] != s[:-1])]           # blocks of equal (user, score)
    blk = np.cumsum(key_new) - 1
    pos = np.bincount(blk, weights=ww * yy); neg = np.bincount(blk, weights=ww * (1 - yy))
    bu = u[key_new]
    ustart = np.r_[True, bu[1:] != bu[:-1]]; ug = np.cumsum(ustart) - 1
    cneg = np.cumsum(neg) - neg
    first = np.flatnonzero(ustart); cneg = cneg - (np.cumsum(neg) - neg)[first][ug]
    wins = pos * (cneg + 0.5 * neg)
    P = np.bincount(ug, weights=pos); N = np.bincount(ug, weights=neg)
    return float(wins.sum() / (P * N).sum())


def video_bootstrap(user, vid, y, scores, draws=500, seed=0):
    """Within-user AUC of each score and of each difference from the first, with 95% intervals from drawing the
    test videos with replacement (each exposure weighted by how often its video is drawn)."""
    vids, inv = np.unique(vid, return_inverse=True); rng = np.random.default_rng(seed)
    names = list(scores); est = {k: weighted_within_user_auc(user, y, scores[k], np.ones(len(y))) for k in names}
    B = {k: np.empty(draws) for k in names}
    for i in range(draws):
        w = np.bincount(rng.integers(0, len(vids), len(vids)), minlength=len(vids))[inv].astype(float)
        for k in names:
            B[k][i] = weighted_within_user_auc(user, y, scores[k], w)
    ci = lambda a: [float(x) for x in np.quantile(a, [0.025, 0.975])]
    out = {k: {"estimate": est[k], "ci95": ci(B[k])} for k in names}
    return out, B
