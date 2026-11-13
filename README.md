# research-harness

A Claude Code skill and project template for running ML research experiments on remote GPU servers
without the usual failure modes: splits that drift, metrics typed by hand, test data leaking into
training, endless tuning of an idea that already failed, and runaway polling loops.

It applies the harness / loop / graph view of agent engineering to experiments:

- **Harness (the space the agent works in).**
  - A frozen protocol: split lists, evaluator and copied baseline numbers, hash-locked and read-only.
  - A claim contract with a numeric pass bar.
  - A launcher that enforces a GPU allow-list and an idle rule.
  - A trace line per event.
  - A PreToolUse hook that blocks edits to frozen files.
- **Loop (the time it keeps trying).**
  - Metrics come only from the frozen evaluator.
  - `gate.py` decides PASS, BUDGET_EXIT (max variants), STALL_EXIT (no gain for N rounds) or REPRO_FAIL (baseline re-implementation too far from the paper).
  - Jobs write a `DONE` file last, so you wait on one file instead of polling.
- **Graph (the route the work follows).** splits → freeze → features → reference run → candidates → gate → write-up, with each step handing the next a file.

## Install the skill

```bash
git clone https://github.com/hardiksharma6555/research-harness-skill ~/.claude/skills/research-harness
```

## Start a project

```bash
cp -r ~/.claude/skills/research-harness/template  /path/to/project      # on the GPU server
cp contract.example.json contract.json   # fill in claim, protocol, gate, compute
python harness/freeze.py add contract.json frozen/eval.py frozen/baselines.json
python harness/launch.py my_run --tag smoke --gpu auto -- python src/train.py ...
python harness/status.py
python harness/gate.py --setting <setting>
```

Then install the guard hook on the machine running Claude Code (see `SKILL.md`).

## Layout

| Path | What |
|---|---|
| `SKILL.md` | How Claude sets up and operates the harness |
| `template/` | Project scaffold: `CLAUDE.md` map, contract example, harness scripts, frozen evaluator |
| `hooks/` | `research_guard.py` PreToolUse hook, config example, settings snippet |
| `examples/aigi/` | Worked example: AI-generated image detection on GenImage + CNNDetection with a GFRE-style CLIP feature-reconstruction detector |

Requirements: Python 3.9+, numpy and scikit-learn for the evaluator. Compute nodes are reached over ssh with
key-based auth; the example code also uses PyTorch and OpenAI CLIP.

The harness / loop / graph framing comes from @0xwhrrari's article "Harness, Loop and Graph: The Enhanced
Edition of Agent Engineering" (Nov 2026).

License: MIT.
