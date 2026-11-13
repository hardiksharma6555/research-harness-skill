#!/usr/bin/env python
"""Feature-reconstruction detector (GFRE-style), single- or multi-layer.

Stage 1: one autoencoder per feature source, trained with L1 loss on FAKE training features only.
Stage 2: an MLP on the standardised residual |f - f_hat| of all sources, trained with BCE on real+fake.
Early stopping on the frozen validation split. Writes $RUN_DIR/scores/<test_set>.npy (probability of
fake, in split order) and $RUN_DIR/train_log.jsonl. It never reads test labels; frozen/eval.py scores it.

Examples:
  GFRE re-implementation (final embedding):  --setting sd14 --source embed
  Multi-layer variant:                       --setting sd14 --source layers --layers 12,16,20,24
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent


def load(feat_dir, source, layers):
    d = Path(feat_dir)
    if not (d / "DONE.json").exists():
        sys.exit(f"train_recon: features not ready in {d}")
    if source == "embed":
        x = np.load(d / "embed.npy", mmap_mode="r")[:, None, :]
    else:
        x = np.load(d / "cls_layers.npy", mmap_mode="r")[:, [l - 1 for l in layers], :]
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def split_labels(name):
    p = ROOT / "frozen" / "splits" / "train" / f"{name}.tsv"
    return torch.tensor([int(l.rsplit("\t", 1)[1]) for l in p.read_text().splitlines() if l.strip()])


class AE(nn.Module):
    def __init__(self, d, h, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class MultiAE(nn.Module):
    def __init__(self, n, d, h, k):
        super().__init__()
        self.aes = nn.ModuleList(AE(d, h, k) for _ in range(n))

    def forward(self, x):  # x: (B, n, d)
        return torch.stack([ae(x[:, i]) for i, ae in enumerate(self.aes)], 1)


def batches(n, bs, shuffle, g=None):
    idx = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def log(rd, **kw):
    with open(rd / "train_log.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["sd14", "progan"])
    ap.add_argument("--features", default="features/clip_l14")
    ap.add_argument("--source", choices=["embed", "layers"], default="embed")
    ap.add_argument("--layers", default="24", help="1-indexed transformer blocks, e.g. 12,16,20,24")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--ae-hidden", type=int, default=512)
    ap.add_argument("--cls-hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rd = Path(os.environ.get("RUN_DIR", ""))
    if not rd.name or not torch.cuda.is_available():
        sys.exit("train_recon: run through harness/launch.py (needs $RUN_DIR and a GPU)")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    dev = "cuda"
    layers = [int(x) for x in a.layers.split(",")]
    fd = ROOT / a.features
    contract = json.loads((ROOT / "contract.json").read_text())
    (rd / "args.json").write_text(json.dumps(vars(a), indent=1))

    xtr = load(fd / f"{a.setting}_train", a.source, layers).to(dev)
    xva = load(fd / f"{a.setting}_val", a.source, layers).to(dev)
    ytr = split_labels(f"{a.setting}_train").float().to(dev)
    yva = split_labels(f"{a.setting}_val").float().to(dev)
    assert len(xtr) == len(ytr) and len(xva) == len(yva), "feature/split length mismatch"
    n_src, d = xtr.shape[1], xtr.shape[2]

    # Stage 1: autoencoder on fake features only
    ftr, fva = xtr[ytr == 1], xva[yva == 1]
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    norm = lambda x: (x - mu) / sd
    ae = MultiAE(n_src, d, a.ae_hidden, a.k).to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=a.lr)
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(a.max_epochs):
        ae.train()
        for b in batches(len(ftr), a.bs, True, g):
            xb = norm(ftr[b])
            loss = (ae(xb) - xb).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        ae.eval()
        with torch.no_grad():
            v = np.mean([(ae(norm(fva[b])) - norm(fva[b])).abs().mean().item() for b in batches(len(fva), 4096, False)])
        log(rd, stage="ae", epoch=ep, val_fake_l1=round(float(v), 5))
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in ae.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    ae.load_state_dict(best_state)
    ae.eval()

    @torch.no_grad()
    def residual(x):
        out = []
        for b in batches(len(x), 4096, False):
            xb = norm(x[b].to(dev))
            out.append((xb - ae(xb)).abs().flatten(1))
        return torch.cat(out)

    rtr, rva = residual(xtr), residual(xva)
    rmu, rsd = rtr.mean(0, keepdim=True), rtr.std(0, keepdim=True) + 1e-6
    rn = lambda r: (r - rmu) / rsd

    # Stage 2: classifier on residuals
    clf = nn.Sequential(nn.Linear(rtr.shape[1], a.cls_hidden), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(a.cls_hidden, 1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = (-1.0, 0.0), None, 0
    for ep in range(a.max_epochs):
        clf.train()
        for b in batches(len(rtr), a.bs, True, g):
            loss = lossf(clf(rn(rtr[b])).squeeze(1), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            logit = clf(rn(rva)).squeeze(1)
            vacc = ((logit > 0).float() == yva).float().mean().item()
            vloss = lossf(logit, yva).item()
        log(rd, stage="cls", epoch=ep, val_acc=round(vacc, 5), val_loss=round(vloss, 5))
        key = (vacc, -vloss)
        if key > best:
            best, best_state, bad = key, {k: t.clone() for k, t in clf.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    clf.load_state_dict(best_state)
    clf.eval()
    train_s = time.time() - t0

    # Score the frozen test sets (no labels read here)
    (rd / "scores").mkdir(exist_ok=True)
    t1, n_scored = time.time(), 0
    for s in contract["protocol"]["test_sets"]:
        x = load(fd / f"test_{s}", a.source, layers)
        with torch.no_grad():
            p = torch.sigmoid(clf(rn(residual(x))).squeeze(1)).cpu().numpy().astype(np.float32)
        np.save(rd / "scores" / f"{s}.npy", p)
        n_scored += len(p)
    log(rd, stage="done", train_seconds=round(train_s, 1),
        head_ms_per_image=round(1000 * (time.time() - t1) / max(n_scored, 1), 4),
        val_acc=best[0], n_sources=n_src)
    print(f"TRAINED setting={a.setting} source={a.source} layers={layers if a.source == 'layers' else 'embed'} "
          f"best_val_acc={best[0]:.4f}")


if __name__ == "__main__":
    main()
