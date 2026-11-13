#!/usr/bin/env python
"""Verify every file listed in FROZEN.sha256 still matches its hash. Exit 1 on any change."""
import sys

from common import ROOT, check_frozen, read_manifest

man = read_manifest()
if not man:
    print("FROZEN EMPTY: nothing frozen yet (run harness/freeze.py add ...)")
    sys.exit(1)
bad = check_frozen()
if bad:
    sys.exit(1)
print(f"FROZEN OK ({len(man)} files) root={ROOT}")
