#!/usr/bin/env python
"""Build split lists for the AI-generated-image benchmark (GenImage + CNNDetection / ForenSynths).

Writes frozen/splits/{train,test}/*.tsv  (one '<abs path>\t<label>' per line, label 1 = fake)
and frozen/splits/MANIFEST.<part>.json with counts. Dataset roots come from paths.json.

Usage: python src/make_splits.py --part sd14|progan|test
Freeze the outputs afterwards: python harness/freeze.py add frozen/splits
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def die(msg):
    sys.exit(f"make_splits: {msg}")


def files_in(d):
    d = Path(d)
    if not d.is_dir():
        die(f"missing directory {d}")
    return sorted(str(p) for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def labeled_tree(root):
    """All images under any '0_real' / '1_fake' directory below root (CNNDetection layout)."""
    root = Path(root)
    if not root.is_dir():
        die(f"missing directory {root}")
    rows = []
    for lab_name, lab in (("0_real", 0), ("1_fake", 1)):
        for d in sorted(root.rglob(lab_name)):
            if d.is_dir():
                rows += [(p, lab) for p in files_in(d)]
    if not rows:
        die(f"no 0_real/1_fake images under {root}")
    return sorted(rows)


def genimage_val(root):
    root = Path(root)
    return [(p, 1) for p in files_in(root / "val" / "ai")] + [(p, 0) for p in files_in(root / "val" / "nature")]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        die(f"{path} exists (frozen splits are never regenerated in place)")
    path.write_text("".join(f"{p}\t{y}\n" for p, y in rows))


def counts(rows):
    c = Counter(y for _, y in rows)
    return {"real": c[0], "fake": c[1], "total": len(rows)}


def part_sd14(paths, proto, seed):
    cfg = proto["settings"]["sd14"]
    root = Path(paths["genimage_sd14_root"])
    pat, bad = paths["sd14_fake_pattern"], paths.get("sd14_exclude_pattern")
    all_fakes = files_in(root / "train" / "ai")
    fakes = [p for p in all_fakes if pat in Path(p).name]
    excluded = len(all_fakes) - len(fakes)
    reals = files_in(root / "train" / "nature")
    rng = random.Random(seed)
    rng.shuffle(fakes)
    rng.shuffle(reals)
    nt, nv = cfg["n_train_per_class"], cfg["n_val_per_class"]
    if len(fakes) < nt + nv or len(reals) < nt + nv:
        die("not enough images for the requested sd14 split sizes")
    train = sorted([(p, 1) for p in fakes[:nt]] + [(p, 0) for p in reals[:nt]])
    val = sorted([(p, 1) for p in fakes[nt:nt + nv]] + [(p, 0) for p in reals[nt:nt + nv]])
    if bad and any(bad in Path(p).name for p, _ in train + val):
        die(f"{bad} files leaked into the sd14 train/val split")
    if {p for p, _ in train} & {p for p, _ in val}:
        die("sd14 train and val overlap")
    out = ROOT / "frozen" / "splits" / "train"
    write(out / "sd14_train.tsv", train)
    write(out / "sd14_val.tsv", val)
    return {"sd14_train": counts(train), "sd14_val": counts(val), "seed": seed,
            "fake_pool": len(fakes), "real_pool": len(reals), "excluded_by_pattern": excluded}


def part_progan(paths, proto, seed):
    train = labeled_tree(paths["progan_train_root"])
    val = labeled_tree(paths["progan_val_root"])
    classes = sorted({Path(p).parent.parent.name for p, _ in train})
    if {p for p, _ in train} & {p for p, _ in val}:
        die("progan train and val overlap")
    out = ROOT / "frozen" / "splits" / "train"
    write(out / "progan_train.tsv", train)
    write(out / "progan_val.tsv", val)
    return {"progan_train": counts(train), "progan_val": counts(val), "classes": classes}


def part_test(paths, proto, seed):
    out = ROOT / "frozen" / "splits" / "test"
    train_files = set()
    for f in (ROOT / "frozen" / "splits" / "train").glob("*.tsv"):
        train_files |= {l.split("\t")[0] for l in f.read_text().splitlines() if l}
    man = {}
    for name in proto["test_sets"]:
        spec = paths["test_sets"][name]
        rows = genimage_val(spec["root"]) if spec["type"] == "genimage_val" else labeled_tree(spec["root"])
        if train_files & {p for p, _ in rows}:
            die(f"test set {name} overlaps a training split")
        write(out / f"{name}.tsv", rows)
        man[name] = counts(rows)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["sd14", "progan", "test"])
    a = ap.parse_args()
    paths = json.loads((ROOT / "paths.json").read_text())
    proto = json.loads((ROOT / "contract.json").read_text())["protocol"]
    seed = proto.get("seed", 0)
    fn = {"sd14": part_sd14, "progan": part_progan, "test": part_test}[a.part]
    man = fn(paths, proto, seed)
    mp = ROOT / "frozen" / "splits" / f"MANIFEST.{a.part}.json"
    if mp.exists():
        die(f"{mp} exists")
    mp.write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
