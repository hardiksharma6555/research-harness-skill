#!/usr/bin/env python
"""Freeze protocol files: record their sha256 in FROZEN.sha256 and make them read-only.

Usage: python harness/freeze.py add PATH [PATH ...]   (directories are expanded)

Append-only by design. A file that is already frozen with the same hash is skipped;
one that changed is an error. Unfreezing is a human decision: edit FROZEN.sha256 by hand.
"""
import os
import stat
import sys
from pathlib import Path

from common import ROOT, die, read_manifest, sha256_file


def expand(paths):
    for p in paths:
        p = (Path.cwd() / p).resolve()
        if ROOT not in [p, *p.parents]:
            die(f"{p} is outside the project root {ROOT}")
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.is_file())
        elif p.is_file():
            yield p
        else:
            die(f"no such file: {p}")


def make_readonly(p):
    mode = os.stat(p).st_mode
    os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "add":
        die(__doc__.strip())
    man = read_manifest()
    new = []
    for p in expand(sys.argv[2:]):
        rel = str(p.relative_to(ROOT))
        if rel == "FROZEN.sha256":
            continue
        h = sha256_file(p)
        if rel in man:
            if man[rel] != h:
                die(f"{rel} is frozen and has CHANGED; a human must resolve this")
            continue
        new.append((h, rel, p))
    if not new:
        print("FREEZE: nothing new")
        return
    mf = ROOT / "FROZEN.sha256"
    if mf.exists():
        os.chmod(mf, os.stat(mf).st_mode | stat.S_IWUSR)
    with open(mf, "a") as f:
        for h, rel, p in new:
            f.write(f"{h}  {rel}\n")
            make_readonly(p)
    make_readonly(mf)
    print(f"FREEZE: added {len(new)} files (total {len(man) + len(new)})")


if __name__ == "__main__":
    main()
