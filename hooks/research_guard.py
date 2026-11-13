#!/usr/bin/env python3
"""Claude Code PreToolUse hook: turns research-harness rules into blocks instead of prose.

Blocks (exit 2, reason on stderr, which Claude sees):
  1. ssh polling loops without -o BatchMode=yes (a dead control socket + retries can lock the account)
  2. write/delete/permission commands that touch frozen protocol files or run evidence of a harness project
  3. rm on dataset roots
  4. running GPU scripts locally instead of through harness/launch.py on a compute node
  5. Edit/Write on frozen files of a harness project

Config: $CLAUDE_PROJECT_DIR/.claude/research_guard.json (see research_guard.example.json).
A human can still do any of these with `! <command>` in the prompt.
"""
import json
import os
import re
import sys

DEFAULT = {
    "project_markers": [],
    "protected": ["frozen/", "contract.json", "FROZEN.sha256", "results.jsonl", "result.json", "traces.jsonl", "runs/"],
    "data_roots": [],
    "gpu_scripts": ["train_recon.py", "extract_features.py"],
}
WRITE_VERBS = r"\b(rm|mv|cp|rsync|chmod|chown|chattr|truncate|tee|dd|unlink|shred)\b|\bsed\s+(-[a-zA-Z]*i|--in-place)|\bperl\s+-[a-zA-Z]*i"


def load_cfg():
    p = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), ".claude", "research_guard.json")
    cfg = dict(DEFAULT)
    try:
        with open(p) as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    return cfg


def block(msg):
    print(f"research_guard: {msg} If a human wants this anyway, they can run it with `! <command>`.", file=sys.stderr)
    sys.exit(2)


def check_bash(cmd, cfg):
    if "ssh" in cmd and re.search(r"\b(while|until|for)\b", cmd) and "BatchMode=yes" not in cmd:
        block("ssh inside a loop must use `-o BatchMode=yes` and stop on exit code 255.")
    in_project = any(m in cmd for m in cfg["project_markers"])
    prot = [p for p in cfg["protected"] if p in cmd]
    if in_project and prot:
        redirect = re.search(r">{1,2}\s*['\"]?[^\s;|&]*(" + "|".join(map(re.escape, prot)) + ")", cmd)
        if redirect or re.search(WRITE_VERBS, cmd):
            block(f"this command writes to protected harness paths ({', '.join(prot)}). Frozen files, results "
                  "and run evidence change only through harness/freeze.py, frozen/eval.py or a human.")
    if re.search(r"\brm\b", cmd) and any(d in cmd for d in cfg["data_roots"]):
        block("deleting dataset directories is not allowed from the agent.")
    if "ssh" not in cmd and re.search(r"\b(python\d?(\.\d+)?|torchrun|accelerate)\b", cmd) \
            and any(s in cmd for s in cfg["gpu_scripts"]):
        block("GPU scripts run on the compute nodes via harness/launch.py, never on this machine.")


def main():
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        sys.exit(0)
    cfg = load_cfg()
    tool, inp = ev.get("tool_name", ""), ev.get("tool_input", {}) or {}
    if tool == "Bash":
        check_bash(inp.get("command", ""), cfg)
    elif tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = inp.get("file_path", "") or inp.get("notebook_path", "")
        if any(m in path for m in cfg["project_markers"]) and any(p in path for p in cfg["protected"]):
            block(f"{path} is a protected harness file.")
    sys.exit(0)


if __name__ == "__main__":
    main()
