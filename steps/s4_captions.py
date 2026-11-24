"""Step 4: can a video with no viewing record be scored from its caption?

Usage: python steps/s4_captions.py <KuaiRand-Pure/data> <pool_captions.parquet> --sample pilot|final

Every model predicts a video's long-view rate from what is known before anyone watches it. Scores are judged on
the test videos' main-feed random exposures.

  metadata          LightGBM on length, category, upload type, age, music type, aspect ratio, caption length
  + frozen text     the same with 32 principal components of a small language model's caption embedding
  + fine-tuned text the small language model (Qwen2.5-0.5B), its last four blocks fine-tuned on the caption, its output added
                    to the metadata model's out-of-fold logit, so it learns what the text adds
  zero-shot judge   a larger instruction-tuned model (Qwen2.5-7B-Instruct) asked whether a viewer would watch the
                    video for 18 seconds, scored by the log-odds of yes; no training. A second prompt asks for a
                    rating from 0 to 9, scored by the expected rating.
  reference         not cold start: the video's long-view rate among the other viewers of its random exposures
"""
import argparse
import json
import sys
import time
from pathlib import Path

import sklearn.ensemble  # noqa: F401  (before lightgbm and torch)
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import lightgbm as lgb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kr import text as kt  # noqa: E402

SMALL, LARGE = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B-Instruct"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LGB = dict(objective="cross_entropy", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
RATING = ("下面是一条快手短视频的信息。{x}\n请判断普通用户刷到这条视频时，愿意认真看下去的可能性有多大，"
          "用0到9的一个数字回答（0=几乎立刻划走，9=一定会看很久）。只回答一个数字。")
JUDGE = ("下面是一条快手短视频的信息。{x}\n一位普通用户刷到这条视频时，会看满18秒（不足18秒的视频则看完）吗？只回答“是”或“否”。")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def fit_lgb(X, v, tr, va, init=None):
    dtr = lgb.Dataset(X[tr], v.rate.values[tr], weight=v.n_exp.values[tr])
    dva = lgb.Dataset(X[va], v.rate.values[va], weight=v.n_exp.values[va])
    m = lgb.train(LGB, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def metadata_models(X, v, tr, va):
    """The metadata model on all train videos, and out-of-fold logits on the train videos for the text model."""
    m = fit_lgb(X, v, tr, va); full = logit(m.predict(X, num_iteration=m.best_iteration))
    oof = full.copy()
    for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
        mm = fit_lgb(X, v, tr[a], va)
        oof[tr[b]] = logit(mm.predict(X[tr[b]], num_iteration=mm.best_iteration))
    return full, oof, m.best_iteration


@torch.no_grad()
def embed(texts, bs=64):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SMALL); mod = AutoModel.from_pretrained(SMALL, torch_dtype=torch.float32).to(DEV).eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=96, return_tensors="pt").to(DEV)
        h = mod(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.concatenate(out)


class Scorer(torch.nn.Module):
    """The small language model with a linear head on the last token; output added to a fixed offset."""
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.lm = AutoModel.from_pretrained(SMALL, torch_dtype=torch.float32)
        self.head = torch.nn.Linear(self.lm.config.hidden_size, 1)
        torch.nn.init.zeros_(self.head.weight); torch.nn.init.zeros_(self.head.bias)

    def forward(self, ids, mask, offset):
        h = self.lm(input_ids=ids, attention_mask=mask).last_hidden_state
        last = mask.sum(1) - 1
        return offset + self.head(h[torch.arange(len(ids)), last]).squeeze(-1)


def finetune(texts, v, offset, tr, va, seed, epochs=3, bs=16, lr=2e-5, top_layers=None, log=print):
    """Binomial loss on each video's long views out of its random exposures; the epoch with the lowest validation
    loss is kept. Returns scores (logits) for every video."""
    from transformers import AutoTokenizer
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(SMALL); tok.padding_side = "right"
    enc = tok(texts, padding=True, truncation=True, max_length=96, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    k = torch.tensor(v.n_long.values, dtype=torch.float32); n = torch.tensor(v.n_exp.values, dtype=torch.float32)
    off = torch.tensor(offset, dtype=torch.float32)
    model = Scorer().to(DEV)
    if top_layers is not None:                        # train only the last `top_layers` blocks and the final norm
        for q in model.lm.parameters():
            q.requires_grad_(False)
        for blk in list(model.lm.layers)[-top_layers:] + [model.lm.norm]:
            for q in blk.parameters():
                q.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": [q for q in model.lm.parameters() if q.requires_grad], "lr": lr}, {"params": model.head.parameters(), "lr": 1e-3}],
                            weight_decay=0.01)

    def loss_of(rows, train):
        model.train(train); tot, den = 0.0, 0.0; order = rows[np.random.permutation(len(rows))] if train else rows
        for i in range(0, len(order), bs):
            j = torch.tensor(order[i:i + bs])
            with torch.set_grad_enabled(train):
                z = model(ids[j].to(DEV), mask[j].to(DEV), off[j].to(DEV))
                kk, nn = k[j].to(DEV), n[j].to(DEV)
                l = -(kk * torch.nn.functional.logsigmoid(z) + (nn - kk) * torch.nn.functional.logsigmoid(-z)).sum()
                if train:
                    opt.zero_grad(); (l / nn.sum()).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(l); den += float(nn.sum())
        return tot / den

    @torch.no_grad()
    def predict():
        model.eval(); out = []
        for i in range(0, len(texts), 64):
            j = torch.arange(i, min(i + 64, len(texts)))
            out.append(model(ids[j].to(DEV), mask[j].to(DEV), off[j].to(DEV)).float().cpu().numpy())
        return np.concatenate(out)

    best, keep = loss_of(va, False), predict()
    log(f"  seed {seed} epoch 0 val loss {best:.5f}")
    for ep in range(1, epochs + 1):
        t = time.time(); trl = loss_of(tr, True); vl = loss_of(va, False)
        log(f"  seed {seed} epoch {ep} train {trl:.5f} val {vl:.5f} ({time.time() - t:.0f}s)")
        if vl < best:
            best, keep = vl, predict()
    del model; torch.mps.empty_cache() if DEV == "mps" else None
    return keep, best


@torch.no_grad()
def judge(texts, bs=16, prompt=None, scale=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LARGE); tok.padding_side = "left"
    mod = AutoModelForCausalLM.from_pretrained(LARGE, torch_dtype=torch.float16).to(DEV).eval()
    yes, no = tok.encode("是", add_special_tokens=False)[0], tok.encode("否", add_special_tokens=False)[0]
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    chats = [tok.apply_chat_template([{"role": "user", "content": (prompt or JUDGE).format(x=x)}], tokenize=False,
                                     add_generation_prompt=True) for x in texts]
    out = []
    for i in range(0, len(chats), bs):
        b = tok(chats[i:i + bs], return_tensors="pt", padding=True).to(DEV)
        lp = torch.log_softmax(mod(**b).logits[:, -1].float(), -1)
        if scale:                                        # expected rating over the digits 0 to 9
            p = torch.softmax(lp[:, digits], -1); out.append((p * torch.arange(10, device=p.device)).sum(-1).cpu().numpy())
        else:
            out.append((lp[:, yes] - lp[:, no]).cpu().numpy())
    del mod; torch.mps.empty_cache() if DEV == "mps" else None
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("captions")
    ap.add_argument("--sample", default="pilot"); a = ap.parse_args()
    der = ROOT / "derived" / a.sample; der.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    v, ex = kt.load(a.data_dir, a.captions, a.sample)
    tr, va, te = (np.flatnonzero(v.part.values == p) for p in ("train", "val", "test"))
    log(f"videos {len(v)} (train {len(tr)}, val {len(va)}, test {len(te)}); exposures {len(ex)}; "
        f"with caption {np.mean(v.text.str.len() > 0):.3f}")
    X = kt.metadata(v); texts = kt.prompt_text(v)

    S = {}                                                  # logit scores for every video
    S["metadata"], oof, it = metadata_models(X, v, tr, va); log(f"metadata trees {it}")

    f = der / "s4_embeddings.npy"
    E = np.load(f) if f.exists() else embed(texts); np.save(f, E)
    Z = PCA(32, random_state=0).fit(E[tr]).transform(E)
    m = fit_lgb(np.column_stack([X, Z]), v, tr, va)
    S["metadata_frozen_text"] = logit(m.predict(np.column_stack([X, Z]), num_iteration=m.best_iteration))

    runs, vls = [], []
    for seed in (0, 1, 2):
        f = der / f"s4_finetuned_seed{seed}.npy"
        if f.exists():
            runs.append(np.load(f)); continue
        z, vl = finetune(texts, v, oof, tr, va, seed, lr=1e-5, top_layers=4, log=log); np.save(f, z); runs.append(z); vls.append(vl)
    S["metadata_finetuned_text"] = np.mean(runs, 0)
    # at prediction time every video gets the metadata model trained on all train videos as its offset
    S["metadata_finetuned_text"] = S["metadata_finetuned_text"] - oof + S["metadata"]

    f = der / "s4_judge.npy"
    J = np.load(f) if f.exists() else judge(texts); np.save(f, J)
    S["zero_shot_judge"] = J
    f = der / "s4_judge_rating.npy"
    J2 = np.load(f) if f.exists() else judge(texts, prompt=RATING, scale=True); np.save(f, J2)
    S["zero_shot_judge_rating"] = J2

    # evaluation on the test videos' random exposures
    row = {x: i for i, x in enumerate(v.video_id.tolist())}
    ex = ex[ex.video_id.map(v.set_index("video_id").part) == "test"].reset_index(drop=True)
    r = ex.video_id.map(row).to_numpy(); user = ex.user_id.to_numpy(); y = ex.long_view.to_numpy().astype(float)
    order = ["metadata", "metadata_frozen_text", "metadata_finetuned_text", "zero_shot_judge", "zero_shot_judge_rating"]
    sc = {k: S[k][r] for k in order}
    # reference, not cold start: the video's long-view rate among the other viewers of its random exposures
    k_, n_ = v.n_long.values[r], v.n_exp.values[r]
    sc["reference_rate_from_other_viewers"] = (k_ - y + 1) / (n_ - 1 + 12)
    auc, B = kt.video_bootstrap(user, ex.video_id.to_numpy(), y, sc)
    diff = lambda k, base: {"estimate": auc[k]["estimate"] - auc[base]["estimate"],
                            "ci95": [float(q) for q in np.quantile(B[k] - B[base], [0.025, 0.975])]}
    w = v.n_exp.values[te]; obs = v.rate.values[te]
    def wcorr(s):
        s = s[te]; ms, mo = np.average(s, weights=w), np.average(obs, weights=w)
        return float(np.average((s - ms) * (obs - mo), weights=w) / np.sqrt(np.average((s - ms) ** 2, weights=w) * np.average((obs - mo) ** 2, weights=w)))
    from scipy.stats import spearmanr
    res = {"sample": a.sample, "videos": {"train": len(tr), "val": len(va), "test": len(te)},
           "test_exposures": int(len(ex)), "test_users": int(len(np.unique(user))),
           "long_view_rate_test": float(y.mean()), "caption_share": float(np.mean(v.text.str.len() > 0)),
           "within_user_auc": auc,
           "difference": {"finetuned_minus_metadata": diff("metadata_finetuned_text", "metadata"),
                          "frozen_minus_metadata": diff("metadata_frozen_text", "metadata"),
                          "finetuned_minus_frozen": diff("metadata_finetuned_text", "metadata_frozen_text"),
                          "metadata_minus_judge": diff("metadata", "zero_shot_judge"),
                          "metadata_minus_judge_rating": diff("metadata", "zero_shot_judge_rating"),
                          "reference_minus_metadata": diff("reference_rate_from_other_viewers", "metadata")},
           "video_level": {k: {"weighted_pearson_with_observed_rate": wcorr(S[k]),
                               "spearman": float(spearmanr(S[k][te], obs).statistic)} for k in order},
           "finetune_best_val_loss": vls}
    res["rule4_met"] = res["difference"]["metadata_minus_judge"]["ci95"][0] > 0
    out = ROOT / "out" / a.sample; out.mkdir(parents=True, exist_ok=True)
    (out / "step4.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
