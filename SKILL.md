---
name: research-harness
description: Set up and operate an experiment harness for ML research projects run by Claude Code on remote GPU servers — a frozen protocol (splits, evaluator, baseline numbers), a claim contract with a pass bar, results produced only by a frozen evaluator, gate-decided loop exits (pass / budget / stall), detached GPU launches with traces, and hooks that block edits to frozen files. Use this whenever starting or running experiments for a paper, reproducing a baseline, launching training or feature-extraction jobs on a GPU cluster, checking whether an idea passed its kill test, or when the user says experiments drift, numbers can't be trusted, or work stalls midway. Pairs with the research-paper skill (which decides WHAT to run); this one decides HOW runs are executed and judged.
---

# Research harness

The model is rarely what breaks in long experiment work. What breaks is the space around it: a split that
silently changed, a number typed by hand, a test set that leaked into training, a sixth variant of an idea
that already failed, a polling loop that locked the account. This skill moves those rules out of prose and
into files, code and hooks, following the harness / loop / graph framing of agent engineering.

## The three layers, mapped to experiments

**Harness (the space).**
- `contract.json`: claim, protocol, gate, compute rules. Frozen.
- `CLAUDE.md`: a short map of the project, not a manual.
- `frozen/`: split lists, `eval.py`, copied baseline numbers. Hash-locked and read-only.
- `harness/launch.py`: the only way to start a run. It enforces the GPU allow-list and idle rule and starts the job detached.
- `traces.jsonl`: one line per event.
- `state.json`: the gate's decision.
- `lessons.md`: mistakes that have cost a fix twice.

**Loop (the time).**
- **L1 verify loop:** `frozen/eval.py` scores every run. It is the only source of metrics.
- **Exits are decided by code** (`harness/gate.py`):
  - **PASS:** the candidate beats the reference by the margin.
  - **BUDGET_EXIT:** max variants reached.
  - **STALL_EXIT:** no gain for N rounds.
  - **REPRO_FAIL:** our baseline re-implementation is too far from the paper.

  The model never declares "done" itself.
- **L4 wait on change:** each job writes `runs/<id>/DONE` last. Wait on that file with one background watcher, not repeated polling.

**Graph (the route).** Kept deliberately small:
splits → freeze → features → reference run (baseline re-implementation) → candidate variants → gate → write-up.
Each arrow is a file contract (a split TSV, a features `DONE.json`, a run's `scores/`).

## Setting up a project

1. Copy `template/` into the project on the server. Copy task code into `src/` (`examples/aigi/` is a complete worked example for AI-generated image detection).
2. Write `contract.json` from `template/contract.example.json`:
   - **claim:** one sentence with a number;
   - **protocol:** exactly as the source paper defines it; list every unspecified detail and the choice made in `open_questions`;
   - **gate:** reference tag, paper value, margin, budget;
   - **compute:** python path, node/GPU allow-list, idle rule.
3. Write `paths.json` (dataset roots) and copy baseline numbers into `frozen/baselines.json`. Parse them from the PDF with a script that re-checks the averages; never retype them by hand.
4. Freeze: `python harness/freeze.py add contract.json frozen/eval.py frozen/baselines.json`.
5. Generate splits through `launch.py`, read the manifest counts, then freeze the splits.
6. Install the guard hook on the machine running Claude Code:
   - copy `hooks/research_guard.py` to `<project>/.claude/hooks/`;
   - write `<project>/.claude/research_guard.json`, using the remote project directory name as a `project_markers` entry;
   - merge `hooks/settings.snippet.json` into `<project>/.claude/settings.json`.

## Operating rules

- **Numbers come only from `result.json` or the gate line.** Quote them; never compute or retype metrics in chat. If a number isn't in a file `eval.py` wrote, it doesn't exist yet.
- **Start runs only through `launch.py`.** It refuses reused run ids, changed frozen files and busy GPUs, and records the code hashes for every run.
- **One run, one question.** Use the run id and tag to say what it tests (`gfre_repro_s0`, `ml_l12-24_k256`). Variants of the candidate idea share the gate's `candidate_tag`. Anything else, such as debugging or smoke tests, uses a different tag so it does not consume budget.
- **Respect the gate.**
  - CONTINUE means you may try another variant.
  - BUDGET_EXIT, STALL_EXIT and REPRO_FAIL mean stop and bring the evidence to the human. Don't patch the idea a sixth time.
- **The frozen protocol never changes mid-project.** If it must change (a bug in a split, a protocol detail learned from the authors), a human edits `FROZEN.sha256` and a new freeze starts. Earlier results stay in `results.jsonl` with their own history.
- **Wait on `DONE` files with a single background watcher.** Every ssh call inside a loop uses `-o BatchMode=yes` and stops on exit code 255.
- **After every run, read `status.py` output**, not raw logs. Open `log.txt` only on failure.
- **When a mistake repeats,** add a line to `lessons.md`, and turn it into a check if possible (an assert in `src/`, a hook rule, a manifest check).

## Reviewer pass (before trusting a PASS)

Spawn one reviewer subagent per lens. Give each one the code and files, not your conclusion:

1. **Leakage:** do train, val and test files overlap? Are excluded patterns really excluded? Do test labels ever reach the training code?
2. **Protocol:** do the splits, test sets, threshold and metric match the source paper and `contract.json`? Are the `open_questions` choices disclosed?
3. **Fair comparison:** is the reference re-implementation within tolerance? Are the copied baseline numbers from the same protocol? Are any rerun baselines marked?

A reviewer is worth exactly as much as the decisions it changes. Each lens returns concrete file/line evidence or nothing.

## Files

```
template/                 copy into a new project
  CLAUDE.md               map (edit the title)
  contract.example.json   claim / protocol / gate / compute
  lessons.md
  harness/                launch.py job.sh gate.py status.py freeze.py check_frozen.py trace.py common.py
  frozen/eval.py          binary-detection evaluator (ACC at threshold + AP per test set)
hooks/                    research_guard.py + example config + settings snippet
examples/aigi/            AI-generated image detection: GenImage + CNNDetection splits, CLIP feature
                          caching, GFRE-style reconstruction detector, GFRE table parser
```

Credit: the harness / loop / graph framing follows @0xwhrrari's "Harness, Loop and Graph" (Nov 2026).
