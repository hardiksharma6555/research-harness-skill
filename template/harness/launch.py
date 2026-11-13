#!/usr/bin/env python
"""Launch one run, detached, on an idle allowed GPU (or CPU-only) of a compute node.

Usage:
  python harness/launch.py RUN_ID --tag TAG [--setting S] [--gpu auto|none|NODE:IDX] -- CMD [ARGS...]

What it does:
  1. refuses if runs/RUN_ID exists or any frozen file changed
  2. picks a GPU from contract.compute.nodes that passes the idle rule (or uses the one given)
  3. writes runs/RUN_ID/{cmd.sh,config.json}, a trace line, then starts harness/job.sh with nohup
The command sees $RUN_DIR. If it writes $RUN_DIR/scores/<test_set>.npy, job.sh runs frozen/eval.py
and harness/gate.py automatically. Wait on runs/RUN_ID/DONE instead of polling the log.
"""
import argparse
import json
import shlex
import subprocess
import sys

from common import ROOT, check_frozen, code_hashes, contract, die, now, read_manifest, trace


def ssh(node, cmd, timeout=60):
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", node, cmd],
                       capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode == 255:
        die(f"ssh to {node} failed: {r.stderr.strip()}")
    return r


def gpu_table(node):
    q = "nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
    r = ssh(node, q)
    rows = []
    for line in r.stdout.strip().splitlines():
        idx, uuid, used, total, util = [x.strip() for x in line.split(",")]
        rows.append(dict(index=int(idx), uuid=uuid, free_gb=(int(total) - int(used)) / 1024, util=int(util)))
    return rows


def pick_gpu(spec, comp):
    nodes = comp["nodes"]
    idle = comp.get("idle", {"max_util": 10, "min_free_gb": 20})
    ok = lambda g: g["util"] <= idle["max_util"] and g["free_gb"] >= idle["min_free_gb"]
    if spec == "none":
        node = next(iter(nodes))
        return node, None, None
    if spec != "auto":
        node, idx = spec.split(":")
        idx = int(idx)
        if idx not in nodes.get(node, []):
            die(f"{node}:{idx} is not in contract.compute.nodes allow-list {nodes}")
        g = next(g for g in gpu_table(node) if g["index"] == idx)
        if not ok(g):
            die(f"{node}:{idx} is busy (util {g['util']}%, free {g['free_gb']:.0f} GB)")
        return node, idx, g["uuid"]
    for node, allowed in nodes.items():
        for g in gpu_table(node):
            if g["index"] in allowed and ok(g):
                return node, g["index"], g["uuid"]
    die("no idle allowed GPU right now; try later or pass --gpu NODE:IDX")


def main():
    if "--" not in sys.argv:
        die(__doc__.strip())
    split = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--setting", default="")
    ap.add_argument("--gpu", default="auto")
    args = ap.parse_args(sys.argv[1:split])
    cmd = sys.argv[split + 1:]
    if not cmd:
        die("empty command")

    rd = ROOT / "runs" / args.run_id
    if rd.exists():
        die(f"{rd} exists; run ids are never reused")
    if not read_manifest():
        die("nothing is frozen yet; freeze the protocol before launching runs")
    if check_frozen():
        die("frozen files changed; a human must resolve this before any run")

    c = contract()
    comp = c["compute"]
    node, idx, uuid = pick_gpu(args.gpu, comp)
    py = comp["python"]
    env = {"PY": py, "CUDA_VISIBLE_DEVICES": uuid or "", **comp.get("node_env", {}).get(node, {})}

    rd.mkdir(parents=True)
    (rd / "cmd.sh").write_text("set -e\n" + shlex.join(cmd) + "\n")
    cfg = dict(run_id=args.run_id, tag=args.tag, setting=args.setting, cmd=cmd, node=node,
               gpu_index=idx, gpu_uuid=uuid, python=py, code=code_hashes(), launched_at=now())
    (rd / "config.json").write_text(json.dumps(cfg, indent=1))
    trace("launch", args.run_id, tag=args.tag, setting=args.setting, node=node, gpu=idx)

    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    # setsid -f fully detaches the job from the ssh session, so ssh returns immediately
    remote = (f"cd {shlex.quote(str(ROOT))} && {exports} setsid -f bash harness/job.sh {shlex.quote(args.run_id)} "
              f"> {shlex.quote(str(rd / 'job.out'))} 2>&1 < /dev/null")
    ssh(node, remote)
    where = f"{node} gpu={idx}" if idx is not None else f"{node} cpu-only"
    print(f"LAUNCHED {args.run_id} tag={args.tag} on {where}; wait for {rd}/DONE")


if __name__ == "__main__":
    main()
