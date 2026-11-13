#!/usr/bin/env python
"""One line per run plus the latest gate lines.  Usage: python harness/status.py [--last N]"""
import argparse
import time

from common import ROOT, check_frozen, load_json, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=15)
    a = ap.parse_args()
    runs = sorted((ROOT / "runs").glob("*/config.json"), key=lambda p: p.stat().st_mtime)[-a.last:]
    for cfg_p in runs:
        rd = cfg_p.parent
        cfg = load_json(cfg_p)
        done = rd / "DONE"
        if done.exists():
            rc = done.read_text().strip()
            state = "ok" if rc == "0" else f"FAILED rc={rc}"
        else:
            age = (time.time() - cfg_p.stat().st_mtime) / 60
            state = f"running {age:.0f}m"
        res = ""
        if (rd / "result.json").exists():
            r = load_json(rd / "result.json")
            res = f" acc={r['mean_acc']:.2f} ap={r['mean_ap']:.2f}"
        where = f"{cfg['node']}:{cfg['gpu_index']}" if cfg.get("gpu_index") is not None else f"{cfg['node']}:cpu"
        print(f"{rd.name:<28} {cfg['tag']:<14} {cfg.get('setting') or '-':<8} {where:<10} {state}{res}")
    st = ROOT / "state.json"
    if st.exists():
        for g in load_json(st).get("gates", {}).values():
            print(g["line"])
    n = len(read_manifest())
    print(f"frozen: {n} files, {'OK' if n and not check_frozen(quiet=True) else 'CHECK FAILED or empty'}")


if __name__ == "__main__":
    main()
