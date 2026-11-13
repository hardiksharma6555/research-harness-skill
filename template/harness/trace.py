#!/usr/bin/env python
"""Append one event to traces.jsonl.  Usage: python harness/trace.py EVENT RUN_ID [key=value ...]"""
import sys

from common import trace

if len(sys.argv) < 3:
    sys.exit(__doc__)
kv = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
trace(sys.argv[1], sys.argv[2], **kv)
