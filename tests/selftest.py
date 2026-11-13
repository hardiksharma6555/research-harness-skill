#!/usr/bin/env python
"""Self-test for the harness template on synthetic data (no GPU, no ssh; needs numpy + scikit-learn).

Builds a throwaway project from template/, then checks: freeze/check_frozen, job.sh -> eval.py -> gate.py,
every gate verdict, and that tampering with frozen files or re-evaluating a run is refused.
Usage: python tests/selftest.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parent.parent
PY = sys.executable
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def sh(root, *args, env=None):
    r = subprocess.run(list(args), cwd=root, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return r.returncode, (r.stdout + r.stderr).strip()


def make_project(tmp, gate_over=None):
    root = Path(tmp) / "proj"
    shutil.copytree(SKILL / "template", root)
    (root / "runs").mkdir()
    rng = np.random.default_rng(0)
    for s in ("a", "b"):
        d = root / "frozen" / "splits" / "test"
        d.mkdir(parents=True, exist_ok=True)
        y = rng.integers(0, 2, 200)
        y[:2] = [0, 1]
        (d / f"{s}.tsv").write_text("".join(f"/img/{s}/{i}.png\t{v}\n" for i, v in enumerate(y)))
    gate = dict(setting="s", metric="mean_acc", reference_tag="ref", reference_paper_value={"s": 70.0},
                reference_tolerance=3.0, candidate_tag="cand", pass_margin=10.0, min_improvement=0.1,
                max_variants=5, stall_rounds=2)
    gate.update(gate_over or {})
    c = dict(project="selftest", claim="x", protocol=dict(threshold=0.5, test_sets=["a", "b"], groups={"g": ["a"]}),
             gate=gate, compute=dict(python=PY, nodes={"localhost": []}))
    (root / "contract.json").write_text(json.dumps(c))
    return root


def labels(root, s):
    return np.array([int(l.rsplit("\t", 1)[1]) for l in (root / "frozen/splits/test" / f"{s}.tsv").read_text().splitlines()])


def run(root, rid, tag, acc, write_scores=True):
    """Create a run whose scores are correct on a fraction `acc` of images, then execute job.sh."""
    rd = root / "runs" / rid
    (rd / "scores").mkdir(parents=True) if write_scores else rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps(dict(run_id=rid, tag=tag, setting="s", node="localhost", gpu_index=None)))
    (rd / "cmd.sh").write_text("echo hello\n")
    if write_scores:
        for s in ("a", "b"):
            y = labels(root, s)
            n_ok = round(acc * len(y))
            p = np.where(y == 1, 0.9, 0.1)
            p[n_ok:] = 1 - p[n_ok:]
            np.save(rd / "scores" / f"{s}.npy", p.astype(np.float32))
    sh(root, "bash", "harness/job.sh", rid, env={"PY": PY})
    return rd


def verdict(root):
    return json.loads((root / "state.json").read_text())["gates"]["s"]["verdict"]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_project(tmp)
        rc, out = sh(root, PY, "harness/check_frozen.py")
        check("check_frozen fails when nothing is frozen", rc != 0)
        rc, out = sh(root, PY, "harness/freeze.py", "add", "contract.json", "frozen")
        check("freeze adds files", rc == 0 and "added" in out, out)
        check("frozen files are read-only", not os.access(root / "contract.json", os.W_OK) or os.geteuid() == 0)
        rc, out = sh(root, PY, "harness/check_frozen.py")
        check("check_frozen passes after freeze", rc == 0, out)
        rc, out = sh(root, PY, "harness/freeze.py", "add", "contract.json")
        check("re-freezing an unchanged file is a no-op", rc == 0 and "nothing new" in out, out)

        rd = run(root, "util1", "util", 0, write_scores=False)
        check("run without scores: DONE=0, no result", (rd / "DONE").read_text().strip() == "0" and not (rd / "result.json").exists())

        rc, out = sh(root, PY, "harness/gate.py", "--setting", "s")
        check("gate: WAITING_FOR_REFERENCE", "WAITING_FOR_REFERENCE" in out, out)

        rd = run(root, "ref1", "ref", 0.70)
        res = json.loads((rd / "result.json").read_text()) if (rd / "result.json").exists() else {}
        check("eval writes result.json with correct accuracy", abs(res.get("mean_acc", -1) - 70.0) < 0.01, (rd / "log.txt").read_text()[-300:])
        check("result.json is read-only", not os.access(rd / "result.json", os.W_OK) or os.geteuid() == 0)
        check("eval reports groups and per-set metrics", "g" in res.get("groups", {}) and set(res.get("per_set", {})) == {"a", "b"})
        check("gate after reference: CONTINUE", verdict(root) == "CONTINUE", verdict(root))
        rc, out = sh(root, PY, "frozen/eval.py", str(rd))
        check("re-evaluating a run is refused", rc != 0 and "already exists" in out, out)
        n_rows = len((root / "results.jsonl").read_text().splitlines())
        check("results.jsonl has exactly one row", n_rows == 1, str(n_rows))

        run(root, "c1", "cand", 0.75)
        check("gate: CONTINUE (+5 < +10)", verdict(root) == "CONTINUE", verdict(root))
        run(root, "c2", "cand", 0.75)
        check("gate: still CONTINUE after 1 stalled round", verdict(root) == "CONTINUE", verdict(root))
        run(root, "c3", "cand", 0.74)
        check("gate: STALL_EXIT after 2 stalled rounds", verdict(root) == "STALL_EXIT", verdict(root))
        run(root, "c4", "cand", 0.85)
        check("gate: PASS at +15", verdict(root) == "PASS", verdict(root))
        rc, out = sh(root, PY, "harness/status.py")
        check("status prints runs and gate line", rc == 0 and "c4" in out and "GATE" in out and "frozen: " in out, out)

        # bad scores are rejected
        rd = run(root, "bad1", "util", 0.5)
        os.chmod(rd / "result.json", 0o644); (rd / "result.json").unlink()
        np.save(rd / "scores" / "a.npy", np.full(7, 0.5, dtype=np.float32))
        rc, out = sh(root, PY, "frozen/eval.py", str(rd))
        check("eval rejects wrong-length scores", rc != 0 and "scores for" in out, out)
        np.save(rd / "scores" / "a.npy", np.full(200, 1.5, dtype=np.float32))
        rc, out = sh(root, PY, "frozen/eval.py", str(rd))
        check("eval rejects out-of-range scores", rc != 0 and "probabilities" in out, out)

        # tampering with a frozen split
        f = root / "frozen/splits/test/a.tsv"
        os.chmod(f, 0o644)
        f.write_text(f.read_text().replace("\t0\n", "\t1\n", 1))
        rc, out = sh(root, PY, "harness/check_frozen.py")
        check("check_frozen detects a changed split", rc != 0 and "a.tsv" in out, out)
        rd = run(root, "after_tamper", "cand", 0.99)
        check("eval refuses to score after tampering", not (rd / "result.json").exists() and (rd / "DONE").read_text().strip() == "97")
        rc, out = sh(root, PY, "harness/launch.py", "x1", "--tag", "t", "--gpu", "none", "--", "true")
        check("launch refuses when frozen files changed", rc != 0 and "frozen files changed" in out, out)
        rc, out = sh(root, PY, "harness/freeze.py", "add", "frozen/splits/test/a.tsv")
        check("freeze refuses a changed frozen file", rc != 0 and "CHANGED" in out, out)

    with tempfile.TemporaryDirectory() as tmp:  # budget exit + repro fail
        root = make_project(tmp, dict(max_variants=3, stall_rounds=9))
        sh(root, PY, "harness/freeze.py", "add", "contract.json", "frozen")
        run(root, "ref1", "ref", 0.70)
        for i, a in enumerate((0.71, 0.72, 0.73)):
            run(root, f"c{i}", "cand", a)
        check("gate: BUDGET_EXIT at max_variants", verdict(root) == "BUDGET_EXIT", verdict(root))
        rc, out = sh(root, PY, "harness/launch.py", "ref1", "--tag", "t", "--gpu", "none", "--", "true")
        check("launch refuses a reused run id", rc != 0 and "never reused" in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        root = make_project(tmp)
        sh(root, PY, "harness/freeze.py", "add", "contract.json", "frozen")
        run(root, "ref1", "ref", 0.60)
        run(root, "c1", "cand", 0.95)
        check("gate: REPRO_FAIL when reference is 10 pts off the paper", verdict(root) == "REPRO_FAIL", verdict(root))

    print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
