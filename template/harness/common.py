"""Shared helpers for the research harness. Standard library only."""
import hashlib
import json
import sys
import time
from pathlib import Path


def find_root(start):
    p = Path(start).resolve()
    for q in [p, *p.parents]:
        if (q / "contract.json").exists():
            return q
    sys.exit(f"research-harness: no contract.json found above {p}")


ROOT = find_root(Path(__file__).parent)


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def contract():
    return load_json(ROOT / "contract.json")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def append_jsonl(path, obj):
    row = {"ts": now(), **obj}
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_manifest():
    """FROZEN.sha256 lines: '<sha256>  <path relative to ROOT>'."""
    man = ROOT / "FROZEN.sha256"
    if not man.exists():
        return {}
    out = {}
    for line in man.read_text().splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            out[rel] = h
    return out


def check_frozen(quiet=False):
    """Return list of frozen files that are missing or changed."""
    bad = []
    for rel, h in read_manifest().items():
        p = ROOT / rel
        if not p.exists() or sha256_file(p) != h:
            bad.append(rel)
    if not quiet and bad:
        print("FROZEN CHANGED: " + ", ".join(bad[:20]), file=sys.stderr)
    return bad


def code_hashes(dirs=("src",)):
    out = {}
    for d in dirs:
        base = ROOT / d
        if base.exists():
            for p in sorted(base.rglob("*.py")):
                out[str(p.relative_to(ROOT))] = sha256_file(p)[:16]
    return out


def trace(event, run_id=None, **kw):
    append_jsonl(ROOT / "traces.jsonl", {"event": event, "run_id": run_id, **kw})


if __name__ == "__main__":
    print(ROOT)
