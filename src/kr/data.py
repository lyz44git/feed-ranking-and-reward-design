"""Load KuaiRand-Pure, assign users to samples and parts, and build the arrays every step uses.

Samples: `pilot` is one fifth of users (user_id % 5 == 0), used only to choose settings and thresholds;
`final` is the other four fifths. Within a sample, users are split into train (50%), validation (10%) and
test (40%). All of a user's events stay in one part.
"""
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np

SEED = 20261124
LOGS = ("log_random_4_22_to_5_08_pure.csv", "log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
HISTORY = 50          # events of history the sequence model reads
MAIN_TAB = 1          # the feed where 99% of random exposures happen
USER_NUM = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
USER_FLAG = ["is_lowactive_period", "is_live_streamer", "is_video_author"]


def sample_filter(sample):
    if sample not in ("pilot", "final"):
        raise ValueError(sample)
    return "user_id % 5 = 0" if sample == "pilot" else "user_id % 5 <> 0"


def split_users(users, seed=SEED):
    """Deterministic 50 / 10 / 40 split of sorted user ids into train, validation and test."""
    users = np.sort(np.asarray(users))
    perm = np.random.default_rng(seed).permutation(len(users))
    part = np.empty(len(users), "U5")
    n_tr, n_va = int(0.5 * len(users)), int(0.1 * len(users))
    part[perm[:n_tr]] = "train"; part[perm[n_tr:n_tr + n_va]] = "val"; part[perm[n_tr + n_va:]] = "test"
    return dict(zip(users.tolist(), part.tolist()))


def load_events(data_dir, sample):
    """All events of the sample's users, sorted by user and time, with the video's basic features."""
    d = Path(data_dir); w = sample_filter(sample); c = duckdb.connect()
    logs = " union all ".join(
        f"select user_id, video_id, time_ms, is_click, long_view, is_like, is_follow, is_hate, play_time_ms, duration_ms, "
        f"is_rand, tab from read_csv_auto('{d / f}') where {w}" for f in LOGS)
    ev = c.sql(f"""
        select l.*, coalesce(try_cast(split_part(v.tag, ',', 1) as int), -1) as tag1,
               (v.video_type = 'AD')::int as ad,
               coalesce(date_diff('day', try_cast(v.upload_dt as date), date '2022-04-22'), 0) as age
        from ({logs}) l left join read_csv_auto('{d / 'video_features_basic_pure.csv'}') v using (video_id)
        order by user_id, time_ms, video_id""").df()
    uf = c.sql(f"select * from read_csv_auto('{d / 'user_features_pure.csv'}') where {w}").df()
    return ev, uf


def group_starts(keys):
    """For sorted keys, the index at which each row's group starts."""
    n = len(keys)
    start = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    return np.repeat(start, np.diff(np.r_[start, n]))


def watch_share(play, dur):
    """Share of the video watched: play time over recorded length, capped at one. A video whose length is recorded as
    zero counts as fully watched once it is played for 0.1 seconds (see steps/s6_sensitivity.py)."""
    return np.clip(play / np.maximum(dur, 0.1), 0, 1)


def prepare(data_dir, sample):
    ev, uf = load_events(data_dir, sample)
    D = SimpleNamespace(sample=sample, n=len(ev))
    D.user = ev.user_id.to_numpy(); D.time = ev.time_ms.to_numpy()
    D.video_codes, vid = np.unique(ev.video_id.to_numpy(), return_inverse=True); D.vid = vid + 1
    D.tag_codes, tag = np.unique(ev.tag1.to_numpy(), return_inverse=True); D.tag = tag + 1
    D.dur = ev.duration_ms.to_numpy() / 1000.0
    D.durb = np.digitize(D.dur, [7, 18, 60, 180]) + 1
    D.play = ev.play_time_ms.to_numpy() / 1000.0
    D.ratio = watch_share(D.play, D.dur)
    for k in ("is_click", "long_view", "is_like", "is_follow", "is_hate", "is_rand", "tab", "ad", "age"):
        setattr(D, k, ev[k].to_numpy())
    D.hour = (D.time // 3_600_000 % 24).astype(np.float32)
    D.ustart = group_starts(D.user)
    D.pos = np.arange(D.n) - D.ustart                              # the user's events before this one
    parts = split_users(np.unique(D.user))
    D.part = np.array([parts[u] for u in D.user.tolist()])
    main = D.tab == MAIN_TAB
    D.random_target = main & (D.is_rand == 1)
    D.standard_target = main & (D.is_rand == 0)
    D.outcomes = {"long_view": D.long_view.astype(np.float32), "valid_play": D.is_click.astype(np.float32),
                  "watch_seconds": D.play.astype(np.float32), "completion": D.ratio.astype(np.float32),
                  "like": D.is_like.astype(np.float32), "hate": D.is_hate.astype(np.float32),
                  "duration": D.dur.astype(np.float32)}
    # what the user did on each past event, read by the sequence model
    D.feedback = np.stack([D.is_click, D.long_view, np.log1p(D.play) / 5, D.ratio, D.is_like, D.is_rand], 1).astype(np.float32)
    prev = np.r_[0, D.time[:-1]]
    D.gap = np.where(D.pos > 0, np.log1p(np.maximum(D.time - prev, 0) / 1000.0), -1.0).astype(np.float32)
    # user features
    one = [x for x in uf.columns if x.startswith("onehot_feat")]
    u = uf.set_index("user_id")
    U = np.column_stack([u[one].to_numpy(np.float32), u[USER_NUM].to_numpy(np.float32), u[USER_FLAG].to_numpy(np.float32)])
    U = np.nan_to_num(U, nan=-1.0)
    row = {uid: i for i, uid in enumerate(u.index.tolist())}
    D.U = U[[row[x] for x in D.user.tolist()]]
    D.n_onehot, D.n_user_num = len(one), len(USER_NUM)
    return D


def idx(D, target, part):
    """Row numbers of target exposures ('random' or 'standard') of the users in a part."""
    mask = D.random_target if target == "random" else D.standard_target
    return np.flatnonzero(mask & (D.part == part))


def item_rate(D, train_idx, y, prior=20):
    """Mean outcome of each video over the training exposures, shrunk toward the overall mean."""
    m = float(y[train_idx].mean()); k = len(D.video_codes) + 1
    s = np.bincount(D.vid[train_idx], weights=y[train_idx], minlength=k)
    c = np.bincount(D.vid[train_idx], minlength=k)
    return ((s + prior * m) / (c + prior)).astype(np.float32)


def history_index(D, rows, length=HISTORY):
    """Row numbers of each target's previous `length` events (oldest first), -1 where the user has fewer."""
    off = np.arange(length, 0, -1)[None, :]
    return np.where(off <= D.pos[rows][:, None], rows[:, None] - off, -1)


def history_aggregates(D):
    """Counts and rates over the user's earlier events, overall and within the target's category.
    These summarize the history without its order: the features of the non-sequence baseline."""
    def before(x, starts):
        cs = np.cumsum(x) - x
        return cs - cs[starts]
    cnt = D.pos.astype(np.float64); cn = np.maximum(cnt, 1)
    cols = [cnt] + [before(x.astype(np.float64), D.ustart) / cn for x in (D.is_click, D.long_view, D.play, D.ratio)]
    order = np.lexsort((D.time, D.tag, D.user))
    key = D.user[order].astype(np.int64) * 10_000 + D.tag[order]
    gs = group_starts(key)
    tcnt = np.empty(D.n); tcnt[order] = np.arange(D.n) - gs; tn = np.maximum(tcnt, 1)
    def before_tag(x):
        xo = x[order].astype(np.float64); cs = np.cumsum(xo) - xo
        out = np.empty(D.n); out[order] = cs - cs[gs]; return out
    cols += [tcnt, before_tag(D.long_view) / tn, before_tag(D.is_click) / tn, before_tag(D.play) / tn, tcnt / cn, D.gap]
    return np.column_stack(cols).astype(np.float32)


def base_features(D, rate):
    """Features of the target video, the time and the user: no history."""
    return np.column_stack([D.dur, D.tag, D.ad, D.age, rate[D.vid], D.hour, D.U]).astype(np.float32)
