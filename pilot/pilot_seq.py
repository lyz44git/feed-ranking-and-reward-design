"""Pilot (one fifth of users only): does a Transformer over the user's history beat non-sequence baselines
on randomly exposed videos? Exploratory; nothing here is a final result."""
import sys, time
import numpy as np
import sklearn.ensemble  # noqa: F401  (import before lightgbm and torch)
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import duckdb
import torch
import torch.nn as nn

d = sys.argv[1]
TARGET = sys.argv[2] if len(sys.argv) > 2 else "long_view"
L = 50
t0 = time.time()
c = duckdb.connect()
logs = " union all ".join(
    f"select user_id,video_id,time_ms,is_click,long_view,is_like,play_time_ms,duration_ms,is_rand,tab from read_csv_auto('{d}/{f}') where user_id%5=0"
    for f in ("log_random_4_22_to_5_08_pure.csv", "log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"))
ev = c.sql(f"""select l.*, coalesce(try_cast(split_part(v.tag,',',1) as int),-1) tag1, (v.video_type='AD')::int ad,
   coalesce(date_diff('day', try_cast(v.upload_dt as date), date '2022-04-22'),0) age
   from ({logs}) l left join read_csv_auto('{d}/video_features_basic_pure.csv') v using(video_id)
   order by user_id, time_ms""").df()
uf = c.sql(f"select * from read_csv_auto('{d}/user_features_pure.csv') where user_id%5=0").df()
print("events", len(ev), "users", ev.user_id.nunique(), f"{time.time()-t0:.0f}s", flush=True)

# encodings
vid_codes, vid = np.unique(ev.video_id.to_numpy(), return_inverse=True); vid = vid + 1
tag_codes, tag = np.unique(ev.tag1.to_numpy(), return_inverse=True); tag = tag + 1
dur = ev.duration_ms.to_numpy() / 1000.0
durb = np.digitize(dur, [7, 18, 60, 180]) + 1
play = ev.play_time_ms.to_numpy() / 1000.0
ratio = np.clip(play / np.maximum(dur, 0.1), 0, 1)
fb = np.stack([ev.is_click, ev.long_view, np.log1p(play) / 5, ratio, ev.is_like, ev.is_rand], 1).astype(np.float32)
user = ev.user_id.to_numpy(); tms = ev.time_ms.to_numpy()
y_all = ev[TARGET].to_numpy().astype(np.float32)
n = len(ev)
start = np.r_[0, np.flatnonzero(user[1:] != user[:-1]) + 1]
ustart = np.repeat(start, np.diff(np.r_[start, n]))
pos = np.arange(n) - ustart                      # events of this user before this one

rng = np.random.default_rng(0)
users = np.unique(user); test_users = set(rng.choice(users, len(users) // 2, replace=False).tolist())
is_test = np.fromiter((u in test_users for u in user), bool, n)
tgt = (ev.is_rand.to_numpy() == 1) & (ev.tab.to_numpy() == 1)
tr_idx = np.flatnonzero(tgt & ~is_test); te_idx = np.flatnonzero(tgt & is_test)
print("targets train", len(tr_idx), "test", len(te_idx), "| history length at target: median", int(np.median(pos[tgt])),
      "p10", int(np.quantile(pos[tgt], .1)), "p90", int(np.quantile(pos[tgt], .9)), flush=True)

# item rate from training users' random exposures (shrunk)
m = y_all[tr_idx].mean()
s = np.bincount(vid[tr_idx], weights=y_all[tr_idx], minlength=len(vid_codes) + 1)
k = np.bincount(vid[tr_idx], minlength=len(vid_codes) + 1)
item_rate = ((s + 20 * m) / (k + 20)).astype(np.float32)

# user features
ucols = [x for x in uf.columns if x.startswith("onehot_feat")] + ["follow_user_num", "fans_user_num", "friend_user_num",
                                                                 "register_days", "is_lowactive_period", "is_live_streamer", "is_video_author"]
ufm = uf.set_index("user_id")[ucols].apply(lambda s_: s_.astype("float32")).fillna(-1)
U = ufm.reindex(user).to_numpy(np.float32)
hour = (ev.time_ms.to_numpy() // 3600000 % 24).astype(np.float32)

# ---------- baselines ----------
def base_feats(idx):
    return np.column_stack([dur[idx], tag[idx], ev.ad.to_numpy()[idx], ev.age.to_numpy()[idx], item_rate[vid[idx]], hour[idx], U[idx]])

def cum_before(x):
    cs = np.cumsum(x) - x
    return cs - cs[ustart]

cnt = pos.astype(np.float32)
h_click = cum_before(fb[:, 0]); h_lv = cum_before(fb[:, 1]); h_play = cum_before(play); h_ratio = cum_before(ratio)
# same-tag history: sort by (user, tag, time)
order = np.lexsort((tms, tag, user)); gkey = user[order].astype(np.int64) * 1000 + tag[order]
gstart = np.r_[0, np.flatnonzero(gkey[1:] != gkey[:-1]) + 1]; gs = np.repeat(gstart, np.diff(np.r_[gstart, n]))
def cum_before_g(x):
    xo = x[order]; cs = np.cumsum(xo) - xo; out = np.empty(n, np.float64); out[order] = cs - cs[gs]; return out
t_cnt = np.empty(n); t_cnt[order] = np.arange(n) - gs
t_lv = cum_before_g(fb[:, 1]); t_click = cum_before_g(fb[:, 0]); t_play = cum_before_g(play)
prev_t = np.r_[0, tms[:-1]]; gap = np.where(pos > 0, np.log1p((tms - prev_t) / 1000.0), -1)

def hist_feats(idx):
    cn = np.maximum(cnt[idx], 1); tn = np.maximum(t_cnt[idx], 1)
    return np.column_stack([cnt[idx], h_click[idx] / cn, h_lv[idx] / cn, h_play[idx] / cn, h_ratio[idx] / cn,
                            t_cnt[idx], t_lv[idx] / tn, t_click[idx] / tn, t_play[idx] / tn, t_cnt[idx] / cn, gap[idx]])

res = {}
def run_lgb(name, Xtr, Xte):
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=47, min_child_samples=50, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    mdl.fit(Xtr, y_all[tr_idx]); p = mdl.predict_proba(Xte)[:, 1]; res[name] = p
    print(f"{name}: AUC {roc_auc_score(y_all[te_idx], p):.4f}", flush=True)

run_lgb("lgb_no_history", base_feats(tr_idx), base_feats(te_idx))
run_lgb("lgb_history_aggregates", np.column_stack([base_feats(tr_idx), hist_feats(tr_idx)]),
        np.column_stack([base_feats(te_idx), hist_feats(te_idx)]))

# ---------- transformer ----------
def hist_index(idx):
    off = np.arange(L, 0, -1)[None, :]                 # oldest .. newest
    h = idx[:, None] - off
    valid = off <= pos[idx][:, None]
    return np.where(valid, h, -1)

NUMS = np.column_stack([np.log1p(dur) / 6, np.log1p(np.maximum(ev.age.to_numpy(), 0)) / 8, ev.ad.to_numpy(), hour / 24]).astype(np.float32)
Un = U.copy(); Un[:, -7:-3] = np.log1p(np.maximum(Un[:, -7:-3], 0)) / 8
Un[:, :-7] = Un[:, :-7] / np.maximum(np.abs(Un[:, :-7]).max(0), 1)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
T = lambda a, dt=None: torch.as_tensor(a, dtype=dt)
G = dict(vid=T(vid, torch.long), tag=T(tag, torch.long), durb=T(durb, torch.long), fb=T(fb), nums=T(NUMS), U=T(Un),
         rate=T(item_rate[vid]), gap=T(np.maximum(gap, 0).astype(np.float32) / 12))

class Net(nn.Module):
    def __init__(s, nv, nt, d=64, seq=True):
        super().__init__(); s.seq = seq
        s.ev, s.et, s.ed = nn.Embedding(nv + 1, 32, padding_idx=0), nn.Embedding(nt + 1, 8, padding_idx=0), nn.Embedding(7, 4, padding_idx=0)
        s.hin = nn.Linear(32 + 8 + 4 + 6 + 1, d); s.tin = nn.Linear(32 + 8 + 4 + 4 + 1, d)
        s.pos = nn.Parameter(torch.zeros(L + 1, d)); nn.init.normal_(s.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        s.uin = nn.Linear(Un.shape[1], 32)
        s.out = nn.Sequential(nn.Linear(d + 32 + d, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(s, idx, h):
        t = torch.cat([s.ev(G["vid"][idx]), s.et(G["tag"][idx]), s.ed(G["durb"][idx]), G["nums"][idx], G["rate"][idx, None]], -1)
        t = s.tin(t); u = torch.relu(s.uin(G["U"][idx]))
        if not s.seq:
            return s.out(torch.cat([t, u, torch.zeros_like(t)], -1)).squeeze(-1)
        pad = h < 0; hh = h.clamp(min=0)
        x = torch.cat([s.ev(G["vid"][hh]), s.et(G["tag"][hh]), s.ed(G["durb"][hh]), G["fb"][hh], G["gap"][hh][..., None]], -1)
        x = s.hin(x) * (~pad)[..., None]
        z = torch.cat([x, t[:, None, :]], 1) + s.pos[None]
        mask = torch.cat([pad, torch.zeros(len(idx), 1, dtype=torch.bool, device=h.device)], 1)
        z = s.enc(z, src_key_padding_mask=mask)
        return s.out(torch.cat([z[:, -1], u, t], -1)).squeeze(-1)

def run_net(name, seq, seed, epochs=4, bs=1024):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(len(vid_codes), len(tag_codes), seq=seq).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Htr, Hte = hist_index(tr_idx), hist_index(te_idx); r = np.random.default_rng(seed)
    yt = torch.as_tensor(y_all)
    for ep in range(epochs):
        net.train(); perm = r.permutation(len(tr_idx)); tot = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; idx = torch.as_tensor(tr_idx[b], device=dev); h = torch.as_tensor(Htr[b], device=dev)
            loss = nn.functional.binary_cross_entropy_with_logits(net(idx, h), yt[tr_idx[b]].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        net.eval(); ps = []
        with torch.no_grad():
            for i in range(0, len(te_idx), 8192):
                ps.append(torch.sigmoid(net(torch.as_tensor(te_idx[i:i + 8192], device=dev), torch.as_tensor(Hte[i:i + 8192], device=dev))).cpu().numpy())
        p = np.concatenate(ps)
        print(f"{name} seed {seed} epoch {ep+1}: train loss {tot/len(perm):.4f} test AUC {roc_auc_score(y_all[te_idx], p):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return p

for k_ in G: G[k_] = G[k_].to(dev)
res["net_no_history"] = run_net("net_no_history", False, 0, epochs=3)
ps = [run_net("transformer", True, sd) for sd in (0, 1)]
res["transformer"] = np.mean(ps, 0)

# user-level bootstrap of AUC differences
yte = y_all[te_idx]; ute = user[te_idx]; uu, inv = np.unique(ute, return_inverse=True)
by = [np.flatnonzero(inv == i) for i in range(len(uu))]
def boot(a, b, B=200):
    r = np.random.default_rng(1); out = []
    for _ in range(B):
        ii = np.concatenate([by[j] for j in r.integers(0, len(uu), len(uu))])
        out.append(roc_auc_score(yte[ii], res[a][ii]) - roc_auc_score(yte[ii], res[b][ii]))
    return np.quantile(out, [.025, .5, .975]).round(4)
print("== final", TARGET)
for k_, p in res.items(): print(k_, round(roc_auc_score(yte, p), 4))
print("transformer - lgb_history_aggregates", boot("transformer", "lgb_history_aggregates"))
print("transformer - lgb_no_history", boot("transformer", "lgb_no_history"))
print("lgb_history_aggregates - lgb_no_history", boot("lgb_history_aggregates", "lgb_no_history"))
for k_, p in res.items():
    top = p >= np.quantile(p, .9)
    print(k_, "top 10%: long_view", round(ev.long_view.to_numpy()[te_idx][top].mean(), 4), "play_s", round(play[te_idx][top].mean(), 2))
