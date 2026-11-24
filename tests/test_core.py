import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kr import data as kd, evaluate as ke  # noqa: E402


def test_split_is_deterministic_and_keeps_shares():
    users = np.arange(1000)
    a, b = kd.split_users(users), kd.split_users(users[::-1])
    assert a == b
    parts = np.array(list(a.values()))
    assert (parts == "train").sum() == 500 and (parts == "val").sum() == 100 and (parts == "test").sum() == 400


def test_pilot_and_final_users_do_not_overlap():
    assert kd.sample_filter("pilot") == "user_id % 5 = 0" and kd.sample_filter("final") == "user_id % 5 <> 0"


def test_history_index_reads_only_earlier_events_of_the_same_user():
    user = np.array([1, 1, 1, 2, 2]); D = SimpleNamespace(pos=np.arange(5) - kd.group_starts(user))
    h = kd.history_index(D, np.array([0, 2, 4]), length=3)
    assert h.tolist() == [[-1, -1, -1], [-1, 0, 1], [-1, -1, 3]]


def test_history_summaries_use_no_current_or_future_outcome():
    rng = np.random.default_rng(0); n = 200
    user = np.sort(rng.integers(0, 10, n)); t = np.arange(n)
    def make(lv):
        D = SimpleNamespace(n=n, user=user, time=t, tag=rng0.integers(1, 4, n), is_click=lv, long_view=lv,
                            play=lv * 10.0, ratio=lv * 1.0, gap=np.zeros(n, np.float32))
        D.ustart = kd.group_starts(user); D.pos = np.arange(n) - D.ustart
        return D
    lv = rng.integers(0, 2, n)
    rng0 = np.random.default_rng(1); A = kd.history_aggregates(make(lv))
    lv2 = lv.copy(); last = np.flatnonzero(np.r_[user[1:] != user[:-1], True]); lv2[last] = 1 - lv2[last]
    rng0 = np.random.default_rng(1); B = kd.history_aggregates(make(lv2))
    assert np.allclose(A, B)        # changing each user's last outcome changes no feature of any row


def test_within_user_auc_matches_sklearn_for_one_user_and_ignores_user_level_shifts():
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 300).astype(float); s = rng.normal(size=300) + y
    assert abs(ke.within_user_auc(np.zeros(300, int), y, s) - roc_auc_score(y, s)) < 1e-12
    user = np.repeat(np.arange(30), 10)
    assert abs(ke.within_user_auc(user, y, s) - ke.within_user_auc(user, y, s + user * 100.0)) < 1e-12


def test_ties_get_half_credit():
    assert ke.within_user_auc(np.zeros(4, int), np.array([0, 1, 0, 1.0]), np.zeros(4)) == 0.5


def test_policy_value_picks_the_top_share_per_user():
    user = np.repeat([0, 1], 10); score = np.tile(np.arange(10.0), 2); out = {"r": np.tile(np.arange(10.0), 2)}
    v, base = ke.policy_value(user, score, out, share=0.2)
    assert v["r"] == 8.5 and base["r"] == 4.5


def test_item_rate_shrinks_toward_the_mean():
    D = SimpleNamespace(video_codes=np.arange(2), vid=np.array([1, 1, 2]))
    r = kd.item_rate(D, np.arange(3), np.array([1.0, 1.0, 0.0]), prior=20)
    assert 2 / 3 < r[1] < 1 and 0 < r[2] < 2 / 3 and abs(r[0] - 2 / 3) < 1e-6
