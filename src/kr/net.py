"""The sequence model: a Transformer encoder over the user's last events, with the candidate video as the last token."""
import time

import numpy as np
import torch
import torch.nn as nn

from . import data as kd


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class SequenceNet(nn.Module):
    def __init__(self, n_videos, n_tags, n_user, n_wide=0, length=kd.HISTORY, d=64, use_history=True):
        super().__init__()
        self.use_history = use_history
        self.video = nn.Embedding(n_videos + 1, 32, padding_idx=0)
        self.tag = nn.Embedding(n_tags + 1, 8, padding_idx=0)
        self.durb = nn.Embedding(7, 4, padding_idx=0)
        self.event_in = nn.Linear(32 + 8 + 4 + 6 + 1, d)       # past event: video, category, length, feedback, gap
        self.target_in = nn.Linear(32 + 8 + 4 + 4 + 1, d)      # candidate: video, category, length, numbers, item rate
        self.position = nn.Parameter(torch.zeros(length + 1, d)); nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.user_in = nn.Linear(n_user, 32)
        self.head = nn.Sequential(nn.Linear(d + 32 + d + n_wide, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, G, rows, hist):
        t = torch.cat([self.video(G["vid"][rows]), self.tag(G["tag"][rows]), self.durb(G["durb"][rows]),
                       G["nums"][rows], G["rate"][rows, None]], -1)
        t = self.target_in(t); u = torch.relu(self.user_in(G["U"][rows]))
        if not self.use_history:
            return self.head(torch.cat([t, u, torch.zeros_like(t), G["wide"][rows]], -1)).squeeze(-1)
        pad = hist < 0; h = hist.clamp(min=0)
        x = torch.cat([self.video(G["vid"][h]), self.tag(G["tag"][h]), self.durb(G["durb"][h]), G["fb"][h], G["gap"][h][..., None]], -1)
        x = self.event_in(x) * (~pad)[..., None]
        z = torch.cat([x, t[:, None, :]], 1) + self.position[None]
        mask = torch.cat([pad, torch.zeros(len(rows), 1, dtype=torch.bool, device=hist.device)], 1)
        z = self.encoder(z, src_key_padding_mask=mask)
        return self.head(torch.cat([z[:, -1], u, t, G["wide"][rows]], -1)).squeeze(-1)


def tensors(D, rate, dev, wide=None):
    U = D.U.copy(); a, b = D.n_onehot, D.n_onehot + D.n_user_num
    U[:, :a] = U[:, :a] / np.maximum(np.abs(U[:, :a]).max(0), 1)
    U[:, a:b] = np.log1p(np.maximum(U[:, a:b], 0)) / 8
    nums = np.column_stack([np.log1p(D.dur) / 6, np.log1p(np.maximum(D.age, 0)) / 8, D.ad, D.hour / 24]).astype(np.float32)
    f = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    return {"vid": f(D.vid, torch.long), "tag": f(D.tag, torch.long), "durb": f(D.durb, torch.long), "fb": f(D.feedback),
            "nums": f(nums), "U": f(U), "rate": f(rate[D.vid]), "gap": f(np.maximum(D.gap, 0) / 12),
            "wide": f(wide if wide is not None else np.zeros((D.n, 0), np.float32))}


def _predict(net, G, D, rows, dev, bs=8192):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            out.append(net(G, torch.as_tensor(r, device=dev), torch.as_tensor(kd.history_index(D, r), device=dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def pairwise_loss(score, y, user):
    """Logistic loss over pairs of exposures of the same user with different labels: the higher label should
    score higher. A user's overall level cancels, so the gradient is about which video, not which user."""
    same = user[:, None] == user[None, :]
    better = (y[:, None] > y[None, :]) & same
    if not better.any():
        return score.sum() * 0
    diff = score[:, None] - score[None, :]
    return nn.functional.softplus(-diff[better]).mean()


def fit_predict(D, train_rows, y, val_rows, eval_rows, loss="bce", use_history=True, seeds=(0, 1, 2), max_epochs=5,
                bs=1024, lr=1e-3, rank_weight=1.0, summaries=False, log=print, name="net"):
    """Train on `train_rows` with labels y (an array over all events) on the label's loss plus `rank_weight`
    times a within-user pairwise loss, stop at the epoch with the lowest validation loss (of the label's loss), and return the mean over seeds of the predictions on each set of rows in `eval_rows`.
    loss: 'bce' for a label in [0, 1]; 'mse' for a real-valued label. Predictions are on the label's scale."""
    dev = device(); t0 = time.time()
    rate = kd.item_rate(D, train_rows, y)
    if loss == "mse":
        rate = rate / max(float(y[train_rows].std()), 1e-6)
    wide = None
    if summaries:                      # the baseline's history summaries, standardized on the training rows
        w = np.column_stack([kd.history_aggregates(D), rate[D.vid]])
        w = np.sign(w) * np.log1p(np.abs(w)); mu, sd = w[train_rows].mean(0), w[train_rows].std(0) + 1e-6
        wide = ((w - mu) / sd).astype(np.float32)
    G = tensors(D, rate, dev, wide)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    crit = nn.functional.binary_cross_entropy_with_logits if loss == "bce" else nn.functional.mse_loss
    H = kd.history_index(D, train_rows)
    ut = torch.as_tensor(D.user, device=dev)
    chunks = [np.arange(i, min(i + bs, len(train_rows))) for i in range(0, len(train_rows), bs)]   # rows are sorted by user
    preds = {k: [] for k in eval_rows}; epochs = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = SequenceNet(len(D.video_codes), len(D.tag_codes), D.U.shape[1], n_wide=G["wide"].shape[1], use_history=use_history).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
        rng = np.random.default_rng(seed); best = (np.inf, None, 0)
        for ep in range(max_epochs):
            net.train()
            for ci in rng.permutation(len(chunks)):          # each batch holds whole runs of users
                b = chunks[ci]; rows = torch.as_tensor(train_rows[b], device=dev)
                sc = net(G, rows, torch.as_tensor(H[b], device=dev))
                l = crit(sc, yt[rows])
                if rank_weight:
                    l = l + rank_weight * pairwise_loss(sc, yt[rows], ut[rows])
                opt.zero_grad(); l.backward(); opt.step()
            pv = torch.as_tensor(_predict(net, G, D, val_rows, dev)); vl = float(crit(pv, torch.as_tensor(y[val_rows], dtype=torch.float32)))
            log(f"{name} seed {seed} epoch {ep + 1}: validation loss {vl:.5f} ({time.time() - t0:.0f}s)")
            if vl < best[0]:
                best = (vl, {k: _predict(net, G, D, r, dev) for k, r in eval_rows.items()}, ep + 1)
        epochs.append(best[2])
        for k in eval_rows:
            p = best[1][k]
            preds[k].append(1 / (1 + np.exp(-p)) if loss == "bce" else p)
    return {k: np.mean(v, 0) for k, v in preds.items()}, {"epochs_kept": epochs, "seconds": round(time.time() - t0)}
