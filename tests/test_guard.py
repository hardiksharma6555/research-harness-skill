#!/usr/bin/env python3
"""Tests for hooks/research_guard.py: pipes synthetic PreToolUse payloads and checks the exit code.
Usage: python3 tests/test_guard.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
HOOK = SKILL / "hooks" / "research_guard.py"
P = "cd ~/myproj && "
CASES = [  # (expected exit code, tool, input)
    (0, "Bash", {"command": "ls"}),
    (2, "Bash", {"command": f'ssh h "{P}rm results.jsonl"'}),
    (2, "Bash", {"command": f'ssh h "{P}echo x >> frozen/baselines.json"'}),
    (2, "Bash", {"command": f'ssh h "{P}ls; chmod u+w contract.json && echo hi"'}),
    (2, "Bash", {"command": f'ssh h "{P}rm -rf runs/c1"'}),
    (2, "Bash", {"command": f'ssh h "{P}sed -i s/a/b/ frozen/eval.py"'}),
    (2, "Bash", {"command": "python src/train_recon.py --setting sd14"}),
    (2, "Bash", {"command": "while true; do ssh h ls; sleep 5; done"}),
    (2, "Bash", {"command": 'ssh h "rm -rf /data/progan"'}),
    (2, "Edit", {"file_path": "/x/myproj/frozen/eval.py"}),
    (2, "Write", {"file_path": "/x/myproj/contract.json"}),
    (0, "Bash", {"command": f'ssh h "rm -rf ~/tmpdir; {P}tail runs/x/log.txt; ls frozen/splits"'}),
    (0, "Bash", {"command": f'ssh h "{P}python3 harness/freeze.py add frozen/splits"'}),
    (0, "Bash", {"command": f'ssh h "{P}cat runs/x/result.json && python3 harness/status.py"'}),
    (0, "Bash", {"command": f'ssh h "{P}cp src/a.py src/b.py && cat runs/c1/result.json"'}),
    (0, "Bash", {"command": "for i in 1 2; do ssh -o BatchMode=yes h ls; done"}),
    (0, "Bash", {"command": f'ssh h "{P}python harness/launch.py r1 --tag x -- python src/train_recon.py"'}),
    (0, "Edit", {"file_path": "/x/myproj/src/train.py"}),
    (0, "Edit", {"file_path": "/x/other/contract.json"}),
]


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        cfg = json.loads((SKILL / "hooks" / "research_guard.example.json").read_text())
        cfg["project_markers"] = ["myproj"]
        cfg["gpu_scripts"] = ["train_recon.py"]
        (Path(tmp) / ".claude").mkdir()
        (Path(tmp) / ".claude" / "research_guard.json").write_text(json.dumps(cfg))
        for want, tool, inp in CASES:
            r = subprocess.run([sys.executable, str(HOOK)], input=json.dumps({"tool_name": tool, "tool_input": inp}),
                               capture_output=True, text=True, env={"CLAUDE_PROJECT_DIR": tmp, "PATH": "/usr/bin:/bin"})
            ok = r.returncode == want
            fails += not ok
            print(("PASS" if ok else "FAIL"), f"want={want} got={r.returncode}", tool, json.dumps(inp)[:90])
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
