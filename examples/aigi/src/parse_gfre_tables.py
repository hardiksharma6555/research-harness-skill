#!/usr/bin/env python
"""Copy the baseline numbers from GFRE (IJCAI-26, paper 0131) Tables 1-2 into frozen/baselines.json.

Both tables are trained on ProGAN (CNNDetection 20-class protocol). Rows are parsed from
`pdftotext -layout`; each must have 19 numbers (18 test sets + average) and the averages are re-checked.
Usage: python src/parse_gfre_tables.py refs/gfre_ijcai26_0131.pdf
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = ["progan", "biggan", "cyclegan", "stargan", "stylegan", "gaugan", "crn", "imle", "seeingdark", "san",
        "deepfake", "adm", "glide", "sd14", "sd15", "vqdm", "midjourney", "wukong"]
METHODS = ["CNNDet", "DCTDet", "Fusing", "LNP", "LGrad", "Univfd", "NPR", "VIB-Net", "DIRE", "AEROBLADE",
           "FakeInversion", "Ours"]


def main():
    txt = subprocess.run(["pdftotext", "-layout", sys.argv[1], "-"], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in txt.splitlines():
        m = re.match(r"^\s*(" + "|".join(map(re.escape, METHODS)) + r")\s+((?:\d+\.\d+\s+){18}\d+\.\d+)\s*$", line)
        if m:
            rows.append((m.group(1), [float(x) for x in m.group(2).split()]))
    if len(rows) != 2 * len(METHODS):
        sys.exit(f"expected {2 * len(METHODS)} table rows, parsed {len(rows)}")
    out = {"source": "He et al., GFRE, IJCAI-26 paper 0131, Table 1 (ACC) and Table 2 (AP)",
           "training": "ProGAN (CNNDetection, 20 classes)", "sets": SETS, "acc": {}, "ap": {}}
    mismatches = []
    for i, (name, vals) in enumerate(rows):
        table = "acc" if i < len(METHODS) else "ap"
        per, avg = dict(zip(SETS, vals[:18])), vals[18]
        calc = round(sum(vals[:18]) / 18, 2)
        key = "GFRE" if name == "Ours" else name
        out[table][key] = {"per_set": per, "avg_printed": avg, "avg_recomputed": calc}
        if abs(calc - avg) > 0.06:
            mismatches.append(f"{table}/{key}: printed {avg} vs mean of 18 columns {calc}")
    out["avg_mismatches"] = mismatches
    out["note"] = "Compare against avg_recomputed (mean of the 18 printed columns); printed averages are kept as-is."
    for m in mismatches:
        print("MISMATCH", m)
    dst = ROOT / "frozen" / "baselines.json"
    if dst.exists():
        sys.exit(f"{dst} exists")
    dst.write_text(json.dumps(out, indent=1))
    gf = out["acc"]["GFRE"], out["ap"]["GFRE"]
    print(f"BASELINES: {len(out['acc'])} methods x 2 tables; GFRE acc {gf[0]['avg_printed']} "
          f"(recomputed {gf[0]['avg_recomputed']}), ap {gf[1]['avg_printed']} (recomputed {gf[1]['avg_recomputed']})")


if __name__ == "__main__":
    main()
