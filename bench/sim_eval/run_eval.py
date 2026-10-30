"""Run the sim eval for a set of benchmark instances against a policy server.

Resolves instances from the released eval set, groups them, applies the canonical
protocol (protocol.yaml: per-task rollout budget and per-task settings), and runs
eval_subtask.py (manipulation) or eval_subtask_nav.py (navigation) once per group.
Run it inside the OmniGibson environment; the policy server can be anywhere the
client can reach (see docs/protocol.md).

--eval-root is a directory of the released data, laid out as <root>/<slice>/<task>/:
    <data>/action/eval/manipulation   slices held_out, prompt_probe
    <data>/action/eval/navigation     slice held_out

Instance selection (first one given wins):
    --config a.json,b.json         explicit instance configs
    --instance <stem>              one instance by file stem, searched under --eval-root
    --slice <slice> [--task <t>]   every instance in that slice (optionally some tasks)

Simulator stacks: the four restaurant_diner navigation tasks (bar, corner_booth,
kitchen_sink, oven) need OmniGibson 3.9.2; everything else needs 3.7.2
(docs/sim_setup.md). Groups that need the other stack than the one installed are
skipped with a message, so run a navigation slice once in each environment.

Examples:
    python bench/sim_eval/run_eval.py --eval-root $DATA/action/eval/manipulation \\
        --slice held_out --task reach_plate --host localhost --port 8000 --out runs/my_method
    python bench/sim_eval/run_eval.py --eval-root $DATA/action/eval/manipulation \\
        --slice held_out --dry-run
    python bench/sim_eval/run_eval.py --eval-root $DATA/action/eval/navigation \\
        --slice held_out --task bar,corner_booth,kitchen_sink,oven --out runs/my_method   # 3.9.2 env

Results land in <out>/<test_task>[_ir]/<slice>/{metrics,videos}/. Score them with
    python bench/sim_eval/score.py --results <out>
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import pathlib
import shlex
import subprocess
import sys
from collections import OrderedDict

import yaml

HERE = pathlib.Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.yaml"

# Scenes that only load on OmniGibson 3.9.x (BEHAVIOR 2026 assets); all other
# instances run on 3.7.x.
SCENES_NEEDING_OG39 = {"restaurant_diner"}


def find_configs(root: pathlib.Path) -> list[pathlib.Path]:
    """Instance configs under root; files and directories starting with "_" are skipped."""
    return sorted(
        p for p in root.rglob("*.json")
        if not p.name.endswith("_tro.json") and not p.name.startswith("README")
        and not any(part.startswith("_") for part in p.relative_to(root).parts)
    )


def resolve_instances(args) -> list[pathlib.Path]:
    if args.config:
        return [pathlib.Path(c.strip()).resolve() for c in args.config.split(",") if c.strip()]
    if args.eval_root is None:
        sys.exit("ERROR: --eval-root is required unless --config is given")
    root = args.eval_root.resolve()
    if args.instance:
        hits = [p for p in find_configs(root) if p.stem == args.instance]
        if len(hits) != 1:
            sys.exit(f"ERROR: --instance {args.instance}: {len(hits)} matches under {root}")
        return hits
    if not args.slice:
        sys.exit("ERROR: select instances with --config, --instance or --slice [--task]")
    tasks = [t.strip() for t in (args.task or "").split(",") if t.strip()] or [""]
    found = []
    for task in tasks:
        sel = root / args.slice / task
        if not sel.is_dir():
            sys.exit(f"ERROR: no directory {sel} (expected <eval-root>/<slice>/<task>/)")
        hits = find_configs(sel)
        if not hits:
            sys.exit(f"ERROR: no instances under {sel}")
        found += hits
    return found


def check_triplet(cfg: pathlib.Path) -> None:
    base = str(cfg)[: -len(".json")]
    for suffix in (".bddl", "_tro.json"):
        if not pathlib.Path(base + suffix).is_file():
            sys.exit(f"ERROR: missing instance file {base + suffix}")


def matches(rule_match: dict, fields: dict) -> bool:
    for key, pats in rule_match.items():
        pats = [pats] if isinstance(pats, str) else pats
        if not any(fnmatch.fnmatchcase(fields.get(key, ""), p) for p in pats):
            return False
    return True


def group_instances(cfgs: list[pathlib.Path]) -> "OrderedDict[tuple, list]":
    """One group per (test task, task type, slice, OmniGibson task, nav budget).

    The protocol depends on the test task and task type; the OmniGibson task splits
    groups because Isaac Sim allows one simulation app per process. The slice only
    names the output directory; it is the eval-set directory the instance sits in
    (<slice>/<task>/<instance>.json), not the config's own "slice" field.
    Navigation instances carry their own `max_steps` and a process applies one
    budget to all its episodes, so each distinct navigation budget gets its own group.
    """
    groups: OrderedDict[tuple, list] = OrderedDict()
    for cfg in cfgs:
        check_triplet(cfg)
        meta = json.loads(cfg.read_text())
        if not meta.get("task_name"):
            sys.exit(f"ERROR: {cfg} has no task_name")
        key = (
            meta.get("test_task_name") or meta["task_name"],
            meta.get("task_type", ""),
            cfg.parent.parent.name,
            meta["task_name"],
            meta.get("max_steps") if meta.get("task_type") == "nav" else None,
        )
        groups.setdefault(key, []).append((cfg, meta))
    return groups


def installed_og_minor() -> tuple[int, int] | None:
    """(major, minor) of the installed OmniGibson, or None if it is not installed."""
    from importlib import metadata
    try:
        return tuple(int(x) for x in metadata.version("omnigibson").split(".")[:2])
    except Exception:
        return None


def needs_og39(meta: dict) -> bool:
    return meta.get("scene_model") in SCENES_NEEDING_OG39


def protocol_for(protocol: dict, test_task: str, task_type: str, first_meta: dict):
    fields = {"test_task": test_task, "task_type": task_type}
    env, applied, script = {}, [], "eval_subtask.py"
    for rule in protocol["rules"]:
        if not matches(rule["match"], fields):
            continue
        applied.append(rule["name"])
        script = rule.get("script", script)
        for k, v in rule.get("env", {}).items():
            env.setdefault(k, str(v))
    if task_type == "nav":
        max_steps = first_meta.get("max_steps")
    else:
        budgets = protocol["max_steps"]
        max_steps = budgets["per_test_task"].get(test_task, budgets["default"])
    return env, applied, script, max_steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_argument_group("instance selection")
    sel.add_argument("--eval-root", type=pathlib.Path, help="released eval set, laid out as <root>/<slice>/<task>/")
    sel.add_argument("--config", help="comma-separated instance config paths")
    sel.add_argument("--instance", help="instance file stem")
    sel.add_argument("--slice", help="e.g. train_sanity, held_out, prompt_probe")
    sel.add_argument("--task", help="with --slice: task directories, comma-separated, e.g. reach_plate,reach_dice")
    ap.add_argument("--host", default="localhost", help="policy server host")
    ap.add_argument("--port", type=int, default=8000, help="policy server port")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/eval"), help="results root")
    ap.add_argument("--no-video", action="store_true", help="skip rollout videos")
    ap.add_argument("--with-ir", action="store_true",
                    help="send privileged target pixel centers with each obs (docs/protocol.md)")
    ap.add_argument("--with-first-frame", action="store_true", help="with --with-ir: also send the frame-0 IR anchor")
    ap.add_argument("--with-wrist-ir", action="store_true", help="with --with-ir: also send wrist-camera centers")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved groups and commands, run nothing")
    args = ap.parse_args()

    protocol = yaml.safe_load(PROTOCOL.read_text())
    groups = group_instances(resolve_instances(args))
    n_inst = sum(len(g) for g in groups.values())
    print(f"{n_inst} instance(s) in {len(groups)} group(s)")

    og = installed_og_minor()
    failed, skipped = 0, []
    for (test_task, task_type, slice_name, og_task, _), members in groups.items():
        env_proto, applied, script, max_steps = protocol_for(protocol, test_task, task_type, members[0][1])
        env = dict(os.environ)
        shown = {}
        for k, v in env_proto.items():
            if k in os.environ:
                shown[k] = f"{os.environ[k]} (from your environment; protocol: {v})"
            else:
                env[k] = v
                shown[k] = v
        if os.environ.get("MAX_STEPS_OVERRIDE"):
            max_steps = os.environ["MAX_STEPS_OVERRIDE"]
            shown["max_steps"] = f"{max_steps} (MAX_STEPS_OVERRIDE)"
        else:
            shown["max_steps"] = str(max_steps)
        env.update({"OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES", "DISPLAY": ""})
        if args.with_wrist_ir:
            env["WITH_WRIST_IR"] = "1"

        log_path = (args.out / f"{test_task}{'_ir' if args.with_ir else ''}" / slice_name).resolve()
        cmd = [
            sys.executable, str(HERE / script),
            "policy=websocket", f"model.host={args.host}", f"model.port={args.port}",
            f"task.name={og_task}", "headless=true",
            f"write_video={'false' if args.no_video else 'true'}",
            f"log_path={log_path}", "partial_scene_load=false",
            "+subtask.config_file='" + ",".join(str(c) for c, _ in members) + "'",
        ]
        if max_steps:
            cmd.append(f"+subtask.max_steps={max_steps}")
        if args.with_ir:
            cmd += ["+with_ir=true", f"+debug_ir_video_dir={log_path / 'ir_debug'}"]
            if args.with_first_frame:
                cmd.append("+with_first_frame=true")

        need = "3.9" if needs_og39(members[0][1]) else "3.7"
        print(f"\n=== {test_task} / {slice_name}  ({len(members)} instance(s), OmniGibson task {og_task})")
        print(f"    script: {script}   rules: {', '.join(applied) or 'none'}   needs OmniGibson {need}.x")
        for k, v in shown.items():
            print(f"    {k} = {v}")
        if og is not None and (og >= (3, 9)) != (need == "3.9"):
            print(f"    SKIPPED: installed OmniGibson is {og[0]}.{og[1]}; run this group in the {need}.x environment")
            skipped.append(test_task)
            continue
        if args.dry_run:
            print("    $ " + shlex.join(cmd))
            continue
        log_path.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL).returncode
        # Judge a group by its per-episode metrics, not the exit code: Isaac Sim
        # can crash during shutdown after every episode has been scored.
        missing = [c.stem for c, _ in members if not (log_path / "metrics" / f"{c.stem}.json").is_file()]
        if missing:
            print(f"    group exited with code {rc}; no metrics for {len(missing)}/{len(members)} "
                  f"instance(s): {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
            failed += 1
        elif rc != 0:
            print(f"    all {len(members)} episode(s) scored; simulator exited with code {rc} on shutdown")

    if skipped:
        tasks = sorted(set(skipped))
        print(f"\nSkipped {len(skipped)} group(s) that need the other OmniGibson stack: {', '.join(tasks)}")
    if not args.dry_run:
        ran = len(groups) - len(skipped)
        print(f"\nDone: {ran - failed}/{ran} group(s) scored every episode. Score with:\n"
              f"    python {HERE / 'score.py'} --results {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
