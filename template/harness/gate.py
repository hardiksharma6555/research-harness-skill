#!/usr/bin/env python
"""Decide the loop exit in code, from results.jsonl and contract.gate. Prints one line, writes state.json.

Verdicts:
  WAITING_FOR_REFERENCE  no reference (re-implemented baseline) run yet
  REPRO_FAIL             our reference is further than the tolerance from the paper's number: human decision
  PASS                   best candidate beats the reference by >= pass_margin
  BUDGET_EXIT            max_variants candidates tried without PASS
  STALL_EXIT             last stall_rounds candidates did not improve the best by >= min_improvement
  CONTINUE               otherwise

Usage: python harness/gate.py [--setting S | --from-run RUN_DIR]
"""
import argparse
import json
from pathlib import Path

from common import ROOT, contract, load_json, now, read_jsonl


def decide(rows, g, setting):
    m = g.get("metric", "mean_acc")
    refs = [r for r in rows if r.get("tag") == g["reference_tag"]]
    cands = [r for r in rows if r.get("tag") == g["candidate_tag"]]
    out = dict(setting=setting, metric=m, n_candidates=len(cands), max_variants=g["max_variants"])
    if not refs:
        out["verdict"] = "WAITING_FOR_REFERENCE"
        return out
    ref = max(refs, key=lambda r: r[m])
    out["reference"] = dict(run_id=ref["run_id"], value=ref[m])
    paper = g.get("reference_paper_value", {})
    paper = paper.get(setting) if isinstance(paper, dict) else paper
    repro_ok = True
    if paper is not None:
        gap = ref[m] - paper
        repro_ok = abs(gap) <= g.get("reference_tolerance", 3.0)
        out["reference"].update(paper=paper, gap=round(gap, 2), ok=repro_ok)
    best, stall = None, 0
    for r in cands:
        if best is None or r[m] >= best[m] + g.get("min_improvement", 0.1):
            best, stall = r, 0
        else:
            stall += 1
    out["stall"] = stall
    if best:
        out["best"] = dict(run_id=best["run_id"], value=best[m], delta=round(best[m] - ref[m], 2))
    if not repro_ok:
        out["verdict"] = "REPRO_FAIL"
    elif best and best[m] - ref[m] >= g["pass_margin"]:
        out["verdict"] = "PASS"
    elif len(cands) >= g["max_variants"]:
        out["verdict"] = "BUDGET_EXIT"
    elif stall >= g["stall_rounds"]:
        out["verdict"] = "STALL_EXIT"
    else:
        out["verdict"] = "CONTINUE"
    return out


def line(o, g):
    s = f"GATE setting={o['setting']} verdict={o['verdict']}"
    if "reference" in o:
        r = o["reference"]
        s += f" | ref {r['run_id']}={r['value']:.2f}"
        if "paper" in r:
            s += f" (paper {r['paper']}, gap {r['gap']:+.2f} {'ok' if r['ok'] else 'TOO FAR'})"
    if "best" in o:
        b = o["best"]
        s += f" | best {b['run_id']}={b['value']:.2f} ({b['delta']:+.2f} / need +{g['pass_margin']})"
    s += f" | variants {o['n_candidates']}/{o['max_variants']}"
    if "stall" in o:
        s += f" stall {o['stall']}/{g['stall_rounds']}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting")
    ap.add_argument("--from-run")
    a = ap.parse_args()
    g = contract()["gate"]
    setting = a.setting
    if a.from_run:
        setting = load_json(Path(a.from_run) / "config.json").get("setting")
    setting = setting or g.get("setting", "")
    rows = [r for r in read_jsonl(ROOT / "results.jsonl") if r.get("setting", "") == setting]
    o = decide(rows, g, setting)
    st_path = ROOT / "state.json"
    st = load_json(st_path) if st_path.exists() else {}
    st.setdefault("gates", {})[setting] = {**o, "line": line(o, g), "at": now()}
    st_path.write_text(json.dumps(st, indent=1, sort_keys=True))
    print(line(o, g))


if __name__ == "__main__":
    main()
