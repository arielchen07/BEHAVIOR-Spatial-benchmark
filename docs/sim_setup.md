# Simulator setup

The sim eval runs on two BEHAVIOR-1K simulator stacks. Install the one(s) for the
tasks you evaluate; each is its own conda environment, separate from your policy
server's (the two only talk over a websocket, `docs/protocol.md`).

| tasks | stack | assets |
|---|---|---|
| all 14 manipulation tasks; navigation to bathroom, bookcase, fridge, garage, table_in_kitchen, table_near_sofa | OmniGibson 3.7.2 (Isaac Sim 4.5) | BEHAVIOR 2025 challenge |
| navigation to bar, corner_booth, kitchen_sink, oven (`restaurant_diner` scene) | OmniGibson 3.9.2 (Isaac Sim 5.1) | BEHAVIOR 2026 challenge |

The diner scene does not load on 3.7.2: its task template expects 2026 assets (a
cabinet there has two doors in the 2026 assets and one in the 2025 ones).

The versions below are the ones every result in the paper was produced with. The
BEHAVIOR-1K installer needs conda and an NVIDIA GPU with RTX cores; see
[Rendering hardware](#rendering-hardware).

## OmniGibson 3.7.2

Use BEHAVIOR-1K commit `839cf641a` (`main`, 18 commits after the `v3.7.2` tag).
Two of those commits matter for this benchmark: a websocket reconnection fix in the
eval client and a fix to the `collecting_childrens_toys` BDDL.

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git && cd BEHAVIOR-1K
git checkout 839cf641a
./setup.sh --new-env --omnigibson --bddl --joylo --eval --dataset
conda activate behavior
pip install pyyaml scipy
```

`--dataset` downloads `behavior-1k-assets` (3.7.2rc1), `omnigibson-robot-assets` and
`2025-challenge-task-instances` into `BEHAVIOR-1K/datasets` (point
`OMNIGIBSON_DATA_PATH` elsewhere to keep them outside the checkout). The paper's
environment: Python 3.10, `isaacsim` 4.5.0.0, torch 2.7.1, numpy 1.26.4.

## OmniGibson 3.9.2 / Isaac Sim 5.1

Use BEHAVIOR-1K tag `v3.9.2`, in a second environment:

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git BEHAVIOR-1K-392 && cd BEHAVIOR-1K-392
git checkout v3.9.2
./setup.sh --new-env behavior392 --omnigibson --bddl --joylo --eval --dataset
conda activate behavior392
pip install "numpy<2" pyyaml scipy hydra-core msgpack websockets
```

JoyLo requires numpy < 2, hence the pin. This downloads `behavior-1k-assets` 3.9.0, `omnigibson-robot-assets` 3.8.2 and
`2026-challenge-task-instances`. The paper's environment: Python 3.11, `isaacsim`
5.1.0.0, torch 2.7.0 (CUDA 12.8). OmniGibson 3.9.2 removed the hydra config tree the
3.7.2 eval used; the harness ships its own copy (`bench/sim_eval/configs/`) and
`og_compat.py` resolves the parts that moved, so one harness drives both stacks.

Two things differ from 3.7.2 and are handled by the harness or worth knowing:

- **Emitter crash on load.** One cabinet in `restaurant_diner` has a rotated,
  non-uniformly scaled frame, and creating its steam/fire emitter fails an
  assertion and kills the process while the scene loads. `eval_subtask_nav.py`
  disables those emitters (a visual effect the eval never reads) before loading.
  Nothing to install.
- **Heavier robot base.** 3.9.2 always sets the R1Pro base link to 250 kg (3.7.2
  did so only through an optional wrapper the 2025 tasks did not use). The diner
  tasks therefore run with heavier base dynamics than the other navigation tasks;
  this is the 2026 challenge's setting and is what the paper's diner numbers used.

Navigation success to a large object (the bar counter is 2.3 × 5.8 m) is measured to
the object's footprint, not its center; that is part of `goal_checker.py`.

## Running

Run `bench/sim_eval/run_eval.py` with the environment's Python. It sets
`OMNIGIBSON_HEADLESS=1` and accepts the Isaac Sim EULA for the child process. It
reads the installed OmniGibson version and skips any group that needs the other
stack (the restaurant_diner instances need 3.9.x, all others 3.7.x), printing which
tasks to run in the other environment. If OmniGibson and JoyLo are not installed as
packages in the environment, set `BEHAVIOR_ROOT` to your BEHAVIOR-1K checkout and
the eval scripts add its `OmniGibson/` and `joylo/` directories to the import path. The
first launch on a machine compiles shaders and takes several minutes; later
launches take one to two. Isaac Sim writes caches under `~/.cache` and temporary
files under `$TMPDIR`: give both several GB, and keep `$TMPDIR` on a local disk
(on a network filesystem Isaac Sim can crash at shutdown, after the episodes have
been scored).

## Rendering hardware

Rendering needs RTX cores and a driver Isaac Sim supports. On our machines (L40S and
RTX A5000) NVIDIA driver 580.82.07 rendered correctly, driver
610.43.02 crashed Isaac Sim on startup, and H200s (no RT cores) produced corrupted
frames. Check `nvidia-smi` before debugging a crash.
