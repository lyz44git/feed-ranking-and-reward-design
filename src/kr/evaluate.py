"""Unbiased evaluation on randomly exposed videos, with intervals from resampling users.

Every random exposure is a uniform draw from the candidate pool, so a user's random exposures are a random
sample of the pool for that user. Ranking them and reading the outcomes of the top-ranked ones estimates what
a policy that shows the user its top-ranked candidates would obtain, without a model of the logging policy.
"""
import numpy as np

TOP_SHARE = 0.10
BOOT = 500


def _groups(user):
    order = np.argsort(user, kind="stable")
    u = user[order]
    start = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
    return order, start, np.r_[start[1:], len(u)]


def user_auc_parts(user, y, score):
    """Per user: (number of positive-negative pairs, pairs ranked correctly). Users with one class contribute 0."""
    order, s, e = _groups(user)
    pairs = np.zeros(len(s)); wins = np.zeros(len(s))
    for g, (a, b) in enumerate(zip(s, e)):
        ii = order[a:b]; yy = y[ii]; npos = yy.sum(); nneg = len(ii) - npos
        if npos == 0 or nneg == 0:
            continue
        r = np.empty(len(ii)); sc = score[ii]; o = np.argsort(sc, kind="stable"); r[o] = np.arange(1, len(ii) + 1)
        # average ranks over ties
        ss = sc[o]; t0 = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]; t1 = np.r_[t0[1:], len(ss)]
        for x0, x1 in zip(t0, t1):
            if x1 - x0 > 1:
                r[o[x0:x1]] = (x0 + 1 + x1) / 2
        pairs[g] = npos * nneg; wins[g] = r[yy == 1].sum() - npos * (npos + 1) / 2
    return pairs, wins


def within_user_auc(user, y, score):
    """Share of (watched, not watched) pairs of the same user that the score orders correctly."""
    p, w = user_auc_parts(user, y, score)
    return float(w.sum() / p.sum())


def top_parts(user, score, outcomes, share=TOP_SHARE):
    """Per user: how many exposures the policy would show (the top `share`, at least one), and the sums of each
    outcome over them. Ties are broken by row order."""
    order, s, e = _groups(user)
    k = np.zeros(len(s)); sums = {name: np.zeros(len(s)) for name in outcomes}
    base = {name: np.zeros(len(s)) for name in outcomes}
    for g, (a, b) in enumerate(zip(s, e)):
        ii = order[a:b]; kk = max(1, int(round(share * len(ii))))
        top = ii[np.argsort(-score[ii], kind="stable")[:kk]]
        k[g] = kk
        for name, v in outcomes.items():
            sums[name][g] = v[top].sum(); base[name][g] = kk * v[ii].mean()
    return k, sums, base


def policy_value(user, score, outcomes, share=TOP_SHARE):
    """Mean of each outcome over the exposures the policy would show, and for showing the same number at random."""
    k, sums, base = top_parts(user, score, outcomes, share)
    return ({n: float(v.sum() / k.sum()) for n, v in sums.items()}, {n: float(v.sum() / k.sum()) for n, v in base.items()})


def bootstrap_ratio_difference(num_a, den_a, num_b, den_b, draws=BOOT, seed=0):
    """95% interval of sum(num_a)/sum(den_a) - sum(num_b)/sum(den_b) when users are resampled."""
    rng = np.random.default_rng(seed); n = len(num_a); out = np.empty(draws)
    for i in range(draws):
        j = rng.integers(0, n, n)
        out[i] = num_a[j].sum() / den_a[j].sum() - num_b[j].sum() / den_b[j].sum()
    est = num_a.sum() / den_a.sum() - num_b.sum() / den_b.sum()
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {"estimate": float(est), "ci95": [float(lo), float(hi)]}


def bootstrap_ratio(num, den, draws=BOOT, seed=0):
    rng = np.random.default_rng(seed); n = len(num); out = np.empty(draws)
    for i in range(draws):
        j = rng.integers(0, n, n); out[i] = num[j].sum() / den[j].sum()
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {"estimate": float(num.sum() / den.sum()), "ci95": [float(lo), float(hi)]}
