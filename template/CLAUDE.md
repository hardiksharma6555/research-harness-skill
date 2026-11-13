# <project> — map

Read this first. It is a map, not a manual: follow the pointers.

- **Claim, protocol, gate, compute rules:** `contract.json` (frozen).
- **Where are we?** `python harness/status.py` (one line per run + gate lines).
- **Launch a run:** `python harness/launch.py RUN_ID --tag TAG --setting S -- <cmd>`; then wait on `runs/RUN_ID/DONE`.
- **Results:** only `frozen/eval.py` produces them (job.sh runs it automatically). Quote `result.json` or the gate line; never type a metric by hand.
- **Exit decision:** `python harness/gate.py --setting S` -> PASS / CONTINUE / BUDGET_EXIT / STALL_EXIT / REPRO_FAIL. BUDGET, STALL and REPRO_FAIL mean stop and talk to a human.
- **Frozen (never edit):** `contract.json`, `frozen/` (splits, eval.py, baselines), `FROZEN.sha256`, `results.jsonl`, `runs/*/result.json`.
- **State and memory:** `state.json` (written by gate.py), `traces.jsonl` (one line per event), `lessons.md` (append a lesson when a mistake costs a fix twice).
- **Code:** `src/`. Data roots: `paths.json`. Cached features: `features/`.
- **Compute:** GPU nodes only, via launch.py (it enforces the allow-list and idle rule).
