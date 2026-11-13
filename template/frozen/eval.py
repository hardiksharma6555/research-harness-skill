#!/usr/bin/env python
"""Frozen evaluator: the only code that turns scores into reported metrics.

Usage: python frozen/eval.py RUN_DIR

Reads RUN_DIR/config.json (tag, setting) and RUN_DIR/scores/<set>.npy (probability of FAKE, one value
per line of frozen/splits/test/<set>.tsv, same order) for every set in contract.protocol.test_sets.
Labels come only from the frozen split files. Writes RUN_DIR/result.json (read-only, once) and appends
a summary to results.jsonl. Self-contained on purpose: it does not import the editable harness code.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent


def die(msg):
    print(f"EVAL ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_frozen():
    man = ROOT / "FROZEN.sha256"
    if not man.exists():
        die("FROZEN.sha256 missing; freeze the protocol first")
    listed = {}
    for line in man.read_text().splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            listed[rel] = h
    for need in ("frozen/eval.py", "contract.json"):
        if need not in listed:
            die(f"{need} is not frozen")
    bad = [rel for rel, h in listed.items() if not (ROOT / rel).exists() or sha(ROOT / rel) != h]
    if bad:
        die("frozen files changed or missing: " + ", ".join(bad[:10]))


def labels(name):
    p = ROOT / "frozen" / "splits" / "test" / f"{name}.tsv"
    if not p.exists():
        die(f"no frozen split for test set {name}")
    return np.array([int(l.rsplit("\t", 1)[1]) for l in p.read_text().splitlines() if l.strip()])


def main():
    if len(sys.argv) != 2:
        die(__doc__.strip())
    run = Path(sys.argv[1]).resolve()
    check_frozen()
    c = json.loads((ROOT / "contract.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    proto = c["protocol"]
    thr = proto.get("threshold", 0.5)
    per, hashes = {}, {}
    for s in proto["test_sets"]:
        f = run / "scores" / f"{s}.npy"
        if not f.exists():
            die(f"missing scores for test set {s}")
        p = np.load(f).astype(np.float64).ravel()
        y = labels(s)
        if len(p) != len(y):
            die(f"{s}: {len(p)} scores for {len(y)} images")
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            die(f"{s}: scores must be finite probabilities in [0, 1]")
        pred = (p > thr).astype(int)
        per[s] = dict(
            acc=round(100 * float((pred == y).mean()), 2),
            ap=round(100 * float(average_precision_score(y, p)), 2),
            real_acc=round(100 * float((pred[y == 0] == 0).mean()), 2),
            fake_acc=round(100 * float((pred[y == 1] == 1).mean()), 2),
            n=int(len(y)),
        )
        hashes[s] = sha(f)[:16]
    sets = proto["test_sets"]
    mean = lambda key, names: round(float(np.mean([per[n][key] for n in names])), 2)
    groups = {g: dict(acc=mean("acc", names), ap=mean("ap", names)) for g, names in proto.get("groups", {}).items()}
    res = dict(run_id=run.name, tag=cfg.get("tag"), setting=cfg.get("setting", ""),
               mean_acc=mean("acc", sets), mean_ap=mean("ap", sets), groups=groups, per_set=per,
               scores_sha256=hashes, evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    out = run / "result.json"
    if out.exists():
        die("result.json already exists; a run is evaluated once")
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    os.chmod(out, 0o444)
    summary = {k: res[k] for k in ("run_id", "tag", "setting", "mean_acc", "mean_ap", "groups")}
    summary["per_set_acc"] = {s: per[s]["acc"] for s in sets}
    with open(ROOT / "results.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": res["evaluated_at"], **summary}, sort_keys=True) + "\n")
    g = " ".join(f"{k}={v['acc']:.2f}/{v['ap']:.2f}" for k, v in groups.items())
    print(f"EVAL {run.name} tag={res['tag']} setting={res['setting']} "
          f"mean_acc={res['mean_acc']:.2f} mean_ap={res['mean_ap']:.2f} {g}")


if __name__ == "__main__":
    main()
