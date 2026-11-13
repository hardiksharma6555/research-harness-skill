#!/usr/bin/env python
"""Cache frozen CLIP ViT-L/14 features for one split list (GPU only).

Writes, in the split's line order:
  OUT/cls_layers.npy  float16 (N, 24, 1024)  CLS token after every transformer block
  OUT/embed.npy       float16 (N, 768)       final image embedding (ln_post + proj), what GFRE uses
  OUT/DONE.json       counts, split sha256, unreadable images

Usage: python src/extract_features.py --split frozen/splits/test/adm.tsv --out features/clip_l14/test_adm
Preprocessing is CLIP's own (bicubic resize of the short side to 224, center crop, CLIP normalisation).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent


class SplitDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return self.tf(im.convert("RGB")), i, True
        except Exception:
            return torch.zeros(3, 224, 224), i, False


@torch.no_grad()
def forward_all(visual, x):
    x = visual.conv1(x.type(visual.conv1.weight.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)  # NLD -> LND (OpenAI CLIP layout)
    outs = []
    for blk in visual.transformer.resblocks:
        x = blk(x)
        outs.append(x[0])
    emb = visual.ln_post(x[0]) @ visual.proj
    return torch.stack(outs, 1), emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="path to ViT-L-14.pt (default: from paths.json)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("extract_features: no GPU visible; run it through harness/launch.py on a GPU node")

    split = (ROOT / a.split) if not Path(a.split).is_absolute() else Path(a.split)
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    if (out / "DONE.json").exists():
        print(f"skip: {out} already done")
        return
    out.mkdir(parents=True, exist_ok=True)
    paths = [l.split("\t")[0] for l in split.read_text().splitlines() if l.strip()]
    weights = a.weights or json.loads((ROOT / "paths.json").read_text())["clip_weights"]
    model, tf = clip.load(weights, device="cuda", jit=False)
    model.eval()
    vis = model.visual
    n, L, W = len(paths), len(vis.transformer.resblocks), vis.transformer.width
    cls_mm = np.lib.format.open_memmap(out / "cls_layers.npy", mode="w+", dtype=np.float16, shape=(n, L, W))
    emb_mm = np.lib.format.open_memmap(out / "embed.npy", mode="w+", dtype=np.float16, shape=(n, vis.output_dim))
    dl = DataLoader(SplitDS(paths, tf), batch_size=a.bs, num_workers=a.workers, pin_memory=True)
    bad, t0, checked = [], time.time(), False
    for x, idx, ok in dl:
        x = x.cuda(non_blocking=True)
        layers, emb = forward_all(vis, x)
        if not checked:  # the manual forward must match CLIP's own encode_image
            ref = model.encode_image(x)
            err = (emb.float() - ref.float()).abs().max().item()
            if err > 5e-2:
                sys.exit(f"extract_features: manual forward differs from encode_image (max err {err})")
            checked = True
        i = idx.numpy()
        cls_mm[i] = layers.float().cpu().numpy().astype(np.float16)
        emb_mm[i] = emb.float().cpu().numpy().astype(np.float16)
        bad += [paths[j] for j, o in zip(i, ok.numpy()) if not o]
    cls_mm.flush()
    emb_mm.flush()
    (out / "bad.txt").write_text("".join(p + "\n" for p in bad))
    if len(bad) > 0.001 * n:
        sys.exit(f"extract_features: {len(bad)} unreadable images (>0.1%), see {out}/bad.txt")
    split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
    done = dict(n=n, layers=L, width=W, embed_dim=vis.output_dim, bad=len(bad), split=str(a.split),
                split_sha256=split_sha, preprocess="clip_default_224", seconds=round(time.time() - t0, 1))
    (out / "DONE.json").write_text(json.dumps(done, indent=1))
    print(f"FEATURES {a.out}: n={n} bad={len(bad)} in {done['seconds']}s")


if __name__ == "__main__":
    main()
