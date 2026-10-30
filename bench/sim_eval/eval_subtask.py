"""Evaluate a policy on the manipulation (reach / place) subtasks.

Runs inside an OmniGibson 3.7.2 environment and connects to a policy server
over the websocket protocol in docs/protocol.md. Use bench/sim_eval/run_eval.py
rather than calling this file directly: the launcher applies the canonical
per-task protocol (bench/sim_eval/protocol.yaml) before starting it.

Each subtask instance is three files from the released eval set:

    {instance}.bddl        — BDDL problem (objects + init + goal)
    {instance}.json        — metadata: task_name, instruction, goal_conditions, ...
    {instance}_tro.json    — initial object/robot poses

Direct usage (Hydra overrides):
    python bench/sim_eval/eval_subtask.py \
        policy=websocket model.host=localhost model.port=8000 \
        task.name=<og_task_name> headless=true write_video=true \
        log_path=runs/example partial_scene_load=false \
        +subtask.config_file=/path/to/instance.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

# Ensure BEHAVIOR-1K is importable (joylo, OmniGibson) BEFORE the heavy imports.
# Optional: a BEHAVIOR-1K source checkout, for envs where OmniGibson and joylo
# are not pip-installed.
BEHAVIOR_ROOT = os.environ.get("BEHAVIOR_ROOT")
if BEHAVIOR_ROOT:
    sys.path.insert(0, os.path.join(BEHAVIOR_ROOT, "OmniGibson"))
    sys.path.insert(0, os.path.join(BEHAVIOR_ROOT, "joylo"))

import hydra  # noqa: E402
import torch as th  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from omnigibson.learning.eval import Evaluator  # noqa: E402
from omnigibson.learning.utils.eval_utils import (  # noqa: E402
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
    generate_basic_environment_config,
)
from omnigibson.learning.utils.obs_utils import create_video_writer  # noqa: E402
from omnigibson.utils.python_utils import recursively_convert_to_torch  # noqa: E402

from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES  # noqa: E402
from gello.robots.sim_robot.og_teleop_utils import (  # noqa: E402
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
    load_available_tasks,
)

# Eval-time IR injector (server-side IR rendering uses these centers).
from privileged_ir.object_center_ir_omnigibson import (  # noqa: E402
    IRObsInjector,
    wrap_policy_forward,
)

# Relative-friendly import of the goal_checker sibling module.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from goal_checker import SubtaskGoalChecker, NavSuccessTracker, LayerReachTracker, ReachDwellTracker  # noqa: E402,I001


# Ensure the privileged IR injector's logger.info messages reach stderr —
# without this, default WARNING level silently drops the per-instance
# "[ir_omnigibson] target instances for IR: ..." and missing-detection
# diagnostics we need to debug IR rendering issues.
logging.getLogger("privileged_ir.object_center_ir_omnigibson").setLevel(logging.INFO)

logger = logging.getLogger("subtask_eval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


# ── Subtask-aware evaluator ──────────────────────────────────────────────

class SubtaskEvaluator(Evaluator):
    """Extends OmniGibson's Evaluator to inject a custom BDDL + TRO."""

    def __init__(self, cfg: DictConfig, bddl_string: str | None = None):
        self._bddl_string = bddl_string
        super().__init__(cfg)

    def load_env(self, env_wrapper: DictConfig):
        """Same as parent but pushes our BDDL string into cfg['task']['predefined_problem']."""
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        if task_name not in available_tasks:
            raise ValueError(f"Unknown task.name: {task_name}")

        task_idx = TASK_NAMES_TO_INDICES[task_name]

        # Human stats (used by the parent for timeout + metrics).
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        eps_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"
        )
        with open(eps_path) as f:
            episodes = [json.loads(line) for line in f]
        for ep in episodes:
            if ep["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats:
                    self.human_stats[k].append(ep[k])
        for k in self.human_stats:
            self.human_stats[k] = (
                sum(self.human_stats[k]) / len(self.human_stats[k])
                if self.human_stats[k] else 0.0
            )

        task_cfg = available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)

        if self._bddl_string is not None:
            cfg["task"]["predefined_problem"] = self._bddl_string
            logger.info("Injected custom BDDL for subtask evaluation")

        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)

        subtask_max_steps = OmegaConf.select(self.cfg, "subtask.max_steps", default=None)
        if self.cfg.max_steps is not None:
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        elif subtask_max_steps is not None:
            cfg["task"]["termination_config"]["max_steps"] = int(subtask_max_steps)
            logger.info(f"Timeout: {subtask_max_steps} steps (from subtask config)")
        else:
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        cfg["task"]["include_obs"] = False

        # ── Scene brightness override (OPT-IN, default = stock behaviour) ────
        # gm.FORCE_LIGHT_INTENSITY is applied by DatasetObject._initialize() to
        # every child Light prim of every dataset object (OmniGibson default
        # 10000). It MUST be set before og.Environment() builds the scene —
        # after that the objects are already initialized.
        #   FORCE_LIGHT_INTENSITY=20000  -> 2x brighter dataset lights
        #   FORCE_LIGHT_INTENSITY=none   -> respect each USD's own intensities
        # Unset (the default) changes nothing, so live runs are unaffected.
        # ⚠ Brightening eval ONLY creates a train/eval appearance mismatch: the
        # training frames were rendered at the stock 10000. Use this to probe
        # whether a task's darkness is a lighting artifact, not as a silent
        # default for results runs.
        _fli = os.environ.get("FORCE_LIGHT_INTENSITY", "").strip()
        if _fli:
            gm.FORCE_LIGHT_INTENSITY = None if _fli.lower() == "none" else float(_fli)
            logger.info(f"[lighting] gm.FORCE_LIGHT_INTENSITY = {gm.FORCE_LIGHT_INTENSITY} "
                        f"(stock default is 10000)")

        env = og.Environment(configs=cfg)
        from hydra.utils import instantiate
        env = instantiate(env_wrapper, env=env)

        # VLM_HEAD_RES: render the HEAD camera above the default 224 so the
        # IR VLM (IR_SOURCE=vlm) can localise small targets in a sharper frame.
        # The RGBLowResWrapper (just applied) forced all cameras to 224; we
        # override ONLY the head here. The server downsamples head->224 for the
        # policy + IR, and the GT projection scales
        # native->224 regardless of render res, so nothing else changes. Wrists
        # stay 224. Unset/224 = original behaviour.
        _head_res = int(os.environ.get("VLM_HEAD_RES", "0") or 0)
        if _head_res and _head_res != 224:
            try:
                from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
                _robot = env.robots[0]
                _sn = ROBOT_CAMERA_NAMES["R1Pro"]["head"].split("::")[1]
                _robot.sensors[_sn].image_height = _head_res
                _robot.sensors[_sn].image_width = _head_res
                env.load_observation_space()
                logger.info(f"[VLM_HEAD_RES] head camera render res -> {_head_res}x{_head_res}")
            except Exception:
                logger.exception("[VLM_HEAD_RES] failed to bump head render res; staying at 224")
        return env

    def load_task_instance_from_tro(self, tro_path: str, open_doors_for_rooms=None,
                                    grasp_target_sim_ids=None) -> None:
        """Load a custom TRO state file instead of the default instance 0."""
        logger.info(f"Loading TRO: {tro_path}")
        with open(tro_path) as f:
            tro_state = recursively_convert_to_torch(json.load(f))

        # Per-instance grasp declaration (generalizable; replaces the FORCE_GRASP_*
        # env vars). Format: "grasps": [{"obj": "<bddl_name>", "arm": "left|right"}].
        # Consumed by the declared-grasp path after the stabilize loop.
        self._declared_grasps = tro_state.get("grasps")

        for tro_key, tro_data in tro_state.items():
            if tro_key == "grasps":
                continue  # not a pose/joint entry; handled after stabilize
            if tro_key == "robot_poses":
                robot_model = self.robot.model_name
                robot_pos = tro_data[robot_model][0]["position"]
                robot_quat = tro_data[robot_model][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                self.env.scene.write_task_metadata(key=tro_key, data=tro_data)
                continue

            if tro_key == "robot_joints":
                # Restore trunk/arm/gripper qpos + qvel to the frame's values so
                # the initial obs and state match what the model saw at train
                # time (otherwise arms default to stow → AbsoluteActions adds
                # the wrong state back to delta predictions).
                robot_model = self.robot.model_name
                joints = tro_data[robot_model][0]
                self._restore_robot_joints(joints)
                continue

            if tro_key not in self.env.task.object_scope:
                logger.warning(f"TRO object {tro_key} not in task scope, skipping")
                continue

            obj = self.env.task.object_scope[tro_key]
            actual = obj.wrapped_obj if hasattr(obj, "wrapped_obj") else obj
            if (
                hasattr(actual, "n_joints")
                and actual.n_joints > 0
                and "joint_pos" not in tro_data
            ):
                # Articulated but we only have root-link pose → set pose only.
                pos = tro_data["root_link"]["pos"]
                ori = tro_data["root_link"]["ori"]
                actual.set_position_orientation(pos, ori)
            else:
                obj.load_state(tro_data, serialized=False)

        # Snapshot object world positions right after the TRO restore (BEFORE the
        # stabilize lets unheld objects fall). Force-grasp uses these to place the
        # target at the demo's NATURAL grasp position — where the forced hand actually
        # holds a tupperware — instead of the eef center. (The live scene is
        # unreliable: OmniGibson's auto-grasp is flaky, so the held object often drops
        # during settle and _ag_obj_in_hand reads empty.)
        self._tro_obj_positions = {}
        for _onm, _oo in list(self.env.task.object_scope.items()):
            try:
                _opos = _oo.get_position_orientation()[0]
                self._tro_obj_positions[_oo.name] = _opos.clone() if hasattr(_opos, "clone") else list(_opos)
            except Exception:  # noqa: BLE001
                pass

        # Stabilize while HOLDING the restored (F0) robot qpos, so the initial
        # observation shows the commanded pose. The old code only zeroed
        # velocities (keep_still) without re-commanding qpos, so an unsupported
        # arm — the aug P0 start backs the robot off the desk, leaving the F0
        # arm pose without desk contact — sagged/oscillated under gravity into a
        # shaky, OOD first obs. Re-pin qpos each physics step. Configurable via
        # STABILIZE_STEPS (default 50; non-aug F0 init is contact-supported and
        # was stable at 25, so this is strictly safer).
        _stab_steps = int(os.environ.get("STABILIZE_STEPS", "50"))
        for _ in range(_stab_steps):
            self._reassert_robot_joints()
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()
            self.robot.keep_still()
        logger.info(f"Stabilized {_stab_steps} steps holding restored qpos")

        # ── Declared-grasp path (generalizable, per-instance; no env vars) ──────
        # The TRO declares which held object goes in which hand:
        #   "grasps": [{"obj": "<bddl_name>", "arm": "left|right"}]
        # We re-seat each object to its PRE-STABILIZE position (self._tro_obj_positions
        # = the demo/baked grasp position = the real finger offset, NOT the eef
        # center) and bond the assisted grasp there. This holds a wide tupperware at
        # its rim and a thin knife at its grasp point with ONE task-agnostic routine,
        # so no FORCE_GRASP_MAX_DIST tuning per task. The env-var branch below stays
        # as a fallback for instances that don't declare "grasps".
        if getattr(self, "_declared_grasps", None):
            try:
                og.sim.step_physics()
                for _arm in self.robot.arm_names:            # clear any stale AG (sim reuse)
                    if self.robot._ag_obj_in_hand.get(_arm) is not None:
                        try:
                            self.robot._release_grasp(arm=_arm)
                        except Exception:  # noqa: BLE001
                            pass
                for _ in range(10):                          # flush OG's multi-step release counter
                    og.sim.step_physics()
                for _g in self._declared_grasps:
                    _bddl = _g.get("obj")
                    _arm = str(_g.get("arm", "")).lower()
                    _obj = self.env.task.object_scope.get(_bddl)
                    if _obj is not None and hasattr(_obj, "wrapped_obj"):
                        _obj = _obj.wrapped_obj
                    if _obj is None:
                        logger.warning(f"Declared-grasp: skip unknown obj={_bddl!r}")
                        continue
                    _pos = getattr(self, "_tro_obj_positions", {}).get(_obj.name)
                    if _arm not in self.robot.arm_names:
                        # 'auto'/unspecified -> nearest FREE eef to the demo grasp position
                        _ref = _pos if _pos is not None else _obj.get_position_orientation()[0]
                        _free = [a for a in self.robot.arm_names
                                 if self.robot._ag_obj_in_hand.get(a) is None]
                        if not _free:
                            logger.warning(f"Declared-grasp: no free arm for {_bddl!r}")
                            continue
                        _arm = min(_free, key=lambda a: float(
                            ((self.robot.eef_links[a].get_position_orientation()[0] - _ref) ** 2).sum() ** 0.5))
                    # snap_to_eef: for objects whose TRO/baked position drifts from the
                    # actual carry-pose eef per instance (carving knife: 0.07-0.50m =>
                    # floating), re-seat ONTO the eef instead of the baked position,
                    # keeping the object's orientation (blade-up). Natural-offset grasps
                    # (tupperware rim) keep the demo/baked position.
                    _ep0 = self.robot.eef_links[_arm].get_position_orientation()[0]
                    _target = _ep0 if _g.get("snap_to_eef") else _pos
                    if _target is not None:
                        _oq = _obj.get_position_orientation()[1]
                        _obj.set_position_orientation(_target, _oq)
                        _obj.keep_still()
                        og.sim.step_physics()
                    _tp = _obj.get_position_orientation()[0]
                    _ep = self.robot.eef_links[_arm].get_position_orientation()[0]
                    _bd = float(((_tp - _ep) ** 2).sum() ** 0.5)
                    self.robot._establish_grasp(arm=_arm, ag_data=(_obj, _obj.root_link), contact_pos=_tp)
                    _ok = self.robot._ag_obj_in_hand.get(_arm) is _obj
                    logger.info(f"Declared-grasp: bonded {_obj.name} -> {_arm} eef "
                                f"({'snapped' if _g.get('snap_to_eef') else 'demo pos'}, "
                                f"obj-eef dist={_bd:.3f}m) {'OK' if _ok else '[WARN establish no-op]'}")
            except Exception as _e:  # noqa: BLE001
                logger.warning(f"Declared-grasp path failed: {_e}")

        # ── Place tasks: re-establish the assisted grasp on the in-hand object ──
        # The TRO restores the gripper qpos CLOSED on the target (the demo's
        # place-start pose) but NOT the AG constraint — the fixed joint that
        # physically holds the object to the gripper. Without it the object slips
        # out under physics and falls (carving's small knife happens to hold; the
        # bigger/boxy tupperware drops within ~0.4 s). Re-create the AG for any arm
        # whose closed gripper is in contact with an object: step physics once so
        # the gripper↔object contact registers, then let OmniGibson's own
        # detection (`_calculate_in_hand_object`) + `_establish_grasp` build the
        # constraint. Gated by FORCE_GRASP_IN_HAND (default OFF → no behaviour
        # change for reach/nav/empty-hand tasks).
        elif os.environ.get("FORCE_GRASP_IN_HAND", "0") not in ("0", "", "false", "False", "no", "NO"):
            # DIRECT AG-attach on the TARGET object only. Re-using OmniGibson's
            # auto-detection (`_calculate_in_hand_object`) is too brittle (needs the
            # gripper "closed enough" by a fixed threshold → fails on wide objects);
            # and bonding the nearest object to EACH eef misbehaved — the free hand
            # grabbed a DISTRACTOR (which then derails the eval), and a target
            # sitting just past the radius was MISSED. So bond ONLY the goal's
            # target object(s) (resolved by sim_id), each to its nearest eef,
            # leaving the other hand empty. A synthesized contact point bypasses the
            # close-enough check + `_find_gripper_contacts`; the policy still
            # releases normally (open gripper → OG AG release). Gated → no effect on
            # reach/nav/empty-hand tasks.
            try:
                og.sim.step_physics()
                # GRASP_ARM natural grasp point: place the target where the demo's
                # FORCED hand actually holds a tupperware (its real finger offset), not
                # the eef center. Use the TRO snapshot (positions before settle) and
                # pick the same-category object nearest the forced eef — reliable even
                # though the live auto-grasp is flaky.
                _forced_arm = os.environ.get("GRASP_ARM", "").strip().lower()
                _natural_pt = None
                _forced_held_obj = None
                if _forced_arm in self.robot.arm_names:
                    _tgt0 = None
                    for _sid in (grasp_target_sim_ids or []):
                        _tgt0 = self.env.scene.object_registry("name", _sid)
                        if _tgt0 is not None:
                            break
                    _cat = getattr(_tgt0, "category", None) if _tgt0 is not None else None
                    _fe = self.robot.eef_links[_forced_arm].get_position_orientation()[0]
                    _bo, _bd = None, 1e9
                    for _nm, _p in getattr(self, "_tro_obj_positions", {}).items():
                        _o = self.env.scene.object_registry("name", _nm)
                        if _o is None or _o is self.robot:
                            continue
                        if _cat is not None and getattr(_o, "category", None) != _cat:
                            continue
                        _d = float(((_p - _fe) ** 2).sum() ** 0.5)
                        if _d < _bd:
                            _bo, _bd, _natural_pt = _nm, _d, _p
                    if _natural_pt is not None and _bd <= 0.5:
                        # Grasp the tupperware the demo ALREADY holds in the forced hand
                        # (THIS object, in place) — NOT the goal object, which on occupied
                        # instances is the in-fridge occupant. Avoids yanking the occupant
                        # out of the fridge AND avoids stacking two tupperwares in one hand.
                        # The layer check is interchangeable, so placing this one still
                        # scores. _targets is overridden to it just below.
                        _forced_held_obj = self.env.scene.object_registry("name", _bo)
                        logger.info(f"GRASP_ARM={_forced_arm}: grasp demo's {_forced_arm}-hand "
                                    f"tupperware {_bo} @ {_bd:.3f}m IN PLACE (leave other tupperwares)")
                    else:
                        _natural_pt = None
                # Clear any STALE assisted-grasp so both arms start FREE. The
                # renderer reuses ONE sim across instances, so a bond from a prior
                # instance persists → both arms read "occupied" → the per-target
                # loop below hits "no free arm" and never (re)grasps the target.
                # (Releasing a free arm is a no-op; a correctly-held target is
                # re-bonded just below.) Also guards env reuse in the eval path.
                for _arm in self.robot.arm_names:
                    if self.robot._ag_obj_in_hand.get(_arm) is not None:
                        try:
                            self.robot._release_grasp(arm=_arm)
                        except Exception:  # noqa: BLE001
                            pass
                _gmax = float(os.environ.get("FORCE_GRASP_MAX_DIST", "0.40"))
                _targets = []
                for _sid in (grasp_target_sim_ids or []):
                    _to = self.env.scene.object_registry("name", _sid)
                    if _to is not None and _to is not self.robot:
                        _targets.append(_to)
                if not _targets:
                    logger.warning(f"Force-grasp: no target object resolved from goal "
                                   f"(sim_ids={grasp_target_sim_ids}) — skipping (object may drop)")
                # GRASP_ARM: override the grasp target to the tupperware the demo holds
                # in the forced hand (grasp the HELD one IN PLACE, not the goal object /
                # in-fridge occupant). _natural_pt is its TRO position → the teleport
                # below re-seats it at the natural grasp point if it slipped during settle.
                if _forced_held_obj is not None:
                    _targets = [_forced_held_obj]
                    logger.info(f"GRASP_ARM: grasp target -> {_forced_held_obj.name} (held); "
                                f"goal object left untouched")
                _target_arms = set()
                # GRASP_ARM (pilot, env-gated): force the target into a SPECIFIC hand
                # so we can randomize which hand holds the tupperware. With the
                # teleport below, the target moves onto that eef and the other hand
                # stays empty (74 is in the fridge) → clean 1-in-hand, chosen hand.
                # Empty/unrecognized → original nearest-free-eef behaviour, no change.
                _forced_arm = os.environ.get("GRASP_ARM", "").strip().lower()
                if _forced_arm and _forced_arm not in self.robot.arm_names:
                    logger.warning(f"GRASP_ARM={_forced_arm!r} not in arm_names="
                                   f"{list(self.robot.arm_names)} — ignoring (using nearest eef)")
                    _forced_arm = ""
                elif _forced_arm:
                    # Force-grasp into a SPECIFIC hand. OmniGibson's AG release is a
                    # MULTI-STEP counter, so right after the stale-AG release above
                    # _ag_obj_in_hand[arm] is NOT yet None — a plain `is None` guard
                    # then mis-fires and the nearest-eef fallback INVERTS the hand
                    # (observed: forced left → bonded right). So fully clear both arms
                    # here (release + step the counter out), then force the arm
                    # UNCONDITIONALLY below (no fallback).
                    logger.info(f"GRASP_ARM={_forced_arm}: forcing target into the {_forced_arm} hand")
                    for _a in self.robot.arm_names:
                        if self.robot._ag_obj_in_hand.get(_a) is not None:
                            try:
                                self.robot._release_grasp(arm=_a)
                            except Exception:  # noqa: BLE001
                                pass
                    for _ in range(10):
                        og.sim.step_physics()
                for _to in _targets:
                    _tp = _to.get_position_orientation()[0]
                    _best_arm, _bestd = None, 1e9
                    if _forced_arm in self.robot.arm_names:
                        # UNCONDITIONAL: always the forced hand (no nearest-eef fallback).
                        _best_arm = _forced_arm
                        _ep0 = self.robot.eef_links[_forced_arm].get_position_orientation()[0]
                        _bestd = float(((_tp - _ep0) ** 2).sum() ** 0.5)
                        _ag_occ = {a: (self.robot._ag_obj_in_hand.get(a) is not None)
                                   for a in self.robot.arm_names}
                        print(f"[GRASP_ARM] {_to.name}: forced->{_best_arm} bestd={_bestd:.3f} "
                              f"ag_occupied={_ag_occ}", flush=True)
                    else:
                        for _arm in self.robot.arm_names:
                            if self.robot._ag_obj_in_hand.get(_arm) is not None:
                                continue  # arm already holding something
                            _eef_pos = self.robot.eef_links[_arm].get_position_orientation()[0]
                            _d = float(((_tp - _eef_pos) ** 2).sum() ** 0.5)
                            if _d < _bestd:
                                _best_arm, _bestd = _arm, _d
                    if _best_arm is None:
                        logger.warning(f"Force-grasp: no free arm for target {_to.name}")
                        continue
                    if _bestd > _gmax:
                        if os.environ.get("FORCE_GRASP_TELEPORT", "0") not in ("0", "", "false", "False", "no", "NO"):
                            # Held_out (prompt-probe) inits leave the target on the
                            # counter, too far to bond (the 59/60 "not bonding"
                            # case). Teleport it ONTO the chosen eef so the bond is
                            # guaranteed (dist ~0); the AG constraint then carries it
                            # with the gripper to whatever (carry) arm pose we set.
                            _ep, _ = self.robot.eef_links[_best_arm].get_position_orientation()
                            # Forced hand: prefer the demo's NATURAL grasp position (where
                            # the other tupperware was held, at the real finger offset)
                            # over the eef center, so the gripper holds the target the way
                            # the demo holds it instead of sitting inside it.
                            _tgt_xyz = _natural_pt if (_forced_arm and _natural_pt is not None) else _ep
                            _oq = _to.get_position_orientation()[1]
                            _to.set_position_orientation(_tgt_xyz, _oq)
                            _to.keep_still()
                            og.sim.step_physics()
                            _tp = _to.get_position_orientation()[0]
                            _bestd = float(((_tp - _ep) ** 2).sum() ** 0.5)
                            logger.info(f"Force-grasp: TELEPORTED {_to.name} onto {_best_arm} eef "
                                        f"(was too far) → dist now {_bestd:.3f}m")
                        else:
                            logger.warning(f"Force-grasp: target {_to.name} is {_bestd:.3f}m from nearest "
                                           f"eef ({_best_arm}) > {_gmax}m — init likely off; not bonding")
                            continue
                    try:
                        self.robot._establish_grasp(
                            arm=_best_arm,
                            ag_data=(_to, _to.root_link),
                            contact_pos=_tp,
                        )
                        _ok = self.robot._ag_obj_in_hand.get(_best_arm) is _to
                        if _ok:
                            _target_arms.add(_best_arm)
                        logger.info(f"Direct-grasp: bonded TARGET {_to.name} to {_best_arm} eef "
                                    f"(dist={_bestd:.3f}m){'' if _ok else ' [WARN: establish no-op]'}")
                    except Exception as _e2:  # noqa: BLE001
                        logger.warning(f"Direct-grasp: _establish_grasp failed for {_to.name} on {_best_arm}: {_e2}")
                # FREE the NON-target hand -- DISABLED by default. This was
                # DROPPING THE TARGET: the force-grasp bonds the AG to the
                # geometrically-nearest eef, which is NOT always the gripper
                # physically holding the target (v1 tupperware: bonded LEFT but
                # held by the RIGHT gripper). Opening the "non-target" hand then
                # opened the REAL holder, the target fell, and the place goal
                # (ontop floor) fired at ~67 steps -> false fail. The checker
                # already scores the SPECIFIC target object, so the 2nd in-hand
                # object is cosmetic and freeing it was never needed. Opt-in via
                # FREE_NONTARGET_HAND=1 (also gates the underground-stash below).
                if _targets and os.environ.get("FREE_NONTARGET_HAND", "0") not in ("0", "", "false", "False", "no", "NO"):
                    for _arm in self.robot.arm_names:
                        if _arm in _target_arms:
                            continue
                        try:
                            if self.robot._ag_obj_in_hand.get(_arm) is not None:
                                self.robot._release_grasp(arm=_arm)
                            _gidx = self.robot.gripper_control_idx[_arm]
                            _open = self.robot.joint_upper_limits[_gidx]
                            self.robot.set_joint_positions(_open, indices=_gidx, drive=False)
                            logger.info(f"Direct-grasp: freed non-target hand {_arm} "
                                        f"(released AG + opened gripper → distractor drops)")
                        except Exception as _e3:  # noqa: BLE001
                            logger.warning(f"Direct-grasp: failed to free hand {_arm}: {_e3}")
                    # DETERMINISTIC distractor removal. Opening the non-target
                    # gripper (above) is FLAKY: the TRO can wedge a same-category
                    # distractor (e.g. the OTHER tupperware) in a gripper with NO
                    # assisted-grasp constraint, so release-AG is a no-op and a
                    # one-shot open does not reliably dislodge it (observed: 2
                    # tupperwares left in hand -> which is the target was ambiguous).
                    # Teleport every NON-target instance of each TARGET's category
                    # far underground. Category-scoped: the target and anything
                    # *inside* it (e.g. a chicken in the tupperware) are untouched.
                    _tgt_names = {getattr(_t, "name", None) for _t in _targets}
                    # Distractor-stash is DISABLED by default. Teleporting the
                    # non-target OUT of the scene is OOD (the policy was trained
                    # seeing both objects) and made it DROP the target (v1
                    # tupperware fell to the floor: 3/4 -> 2/4). The checker scores
                    # the SPECIFIC target (cond['obj']), already placed correctly,
                    # so the 2nd in-hand object is cosmetic and never affected
                    # scoring. A non-OOD in-scene drop would be needed to free it
                    # for clean videos; the underground teleport is opt-in only.
                    if os.environ.get("STASH_DISTRACTOR", "0") not in ("0", "", "false", "False", "no", "NO"):
                        _tgt_cats = {getattr(_t, "category", None) for _t in _targets}
                    else:
                        _tgt_cats = set()
                    _tgt_cats.discard(None)
                    _stashed = 0
                    for _ent in self.env.task.object_scope.values():
                        _obj = _ent.wrapped_obj if hasattr(_ent, "wrapped_obj") else _ent
                        if _obj is None or _obj is self.robot:
                            continue
                        if getattr(_obj, "name", None) in _tgt_names:
                            continue
                        if getattr(_obj, "category", None) not in _tgt_cats:
                            continue
                        try:
                            for _arm in self.robot.arm_names:
                                if self.robot._ag_obj_in_hand.get(_arm) is _obj:
                                    self.robot._release_grasp(arm=_arm)
                            _q = _obj.get_position_orientation()[1]
                            _obj.set_position_orientation(
                                th.tensor([100.0, 100.0, -50.0], dtype=th.float32), _q)
                            try:
                                _obj.keep_still()
                            except Exception:  # noqa: BLE001
                                pass
                            _stashed += 1
                            logger.info(f"Distractor-stash: teleported non-target "
                                        f"{_obj.name} (cat={getattr(_obj, 'category', None)}) underground")
                        except Exception as _e5:  # noqa: BLE001
                            logger.warning(f"Distractor-stash: failed for "
                                           f"{getattr(_obj, 'name', '?')}: {_e5}")
                    if _stashed:
                        og.sim.step_physics()
            except Exception as _e:  # noqa: BLE001
                logger.warning(f"Direct-grasp block failed: {_e}")

        # Phase B (env-gated): controller-held settle. The pin-stabilize above
        # teleports the joints to F0 every step, so the robot's PD+gravity-comp
        # controllers never actually have to hold the pose under physics; at
        # rollout start they then must produce gravity-comp torques cold while
        # the policy also issues its first action -- that's the residual
        # handoff transient. Phase B drops the teleport for N more steps,
        # forcing the controllers to actually equilibrate at F0 (keep_still
        # re-targets the held pose + zeros vels each step). Default 0 = OFF
        # (no behaviour change); set STAB_CTRL_STEPS=30 to enable ~0.5 s of
        # controller-only settle before snapshotting initial_file.
        _ctrl_steps = int(os.environ.get("STAB_CTRL_STEPS", "0"))
        for _ in range(_ctrl_steps):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()
            self.robot.keep_still()
        if _ctrl_steps:
            logger.info(f"Phase B: controller-held settle {_ctrl_steps} steps")

        # ── Nav precondition: open the doors bounding the goal room ───────────
        # Doors are building STRUCTURE (force-loaded in both partial and full
        # scene mode) but NOT BDDL task objects, so object_scope never includes
        # them and the TRO never touches them — they re-instantiate CLOSED from
        # the cached scene template, physically blocking a move_to_<room> nav
        # even when the policy navigates correctly (the demos had the path doors
        # OPEN). Open every door whose `in_rooms` touches the goal room HERE,
        # before the initial-file snapshot so it survives env.reset(). The goal
        # room(s) come from the in_room goal_conditions; OPEN_DOOR_NAMES /
        # OPEN_DOOR_CATEGORIES (comma lists) are optional extra overrides. No-op
        # for non-nav tasks (no goal room, no overrides) → manipulation evals
        # are unaffected.
        _door_rooms = set(open_doors_for_rooms or [])
        # Per-instance door declaration (generalizable; preferred over the
        # OPEN_DOOR_* env vars — same pattern as the TRO "grasps" key). Format:
        #   "open_doors": {"categories": [...], "names": [...], "joint_names": [...]}
        # Needed for e.g. place_bowl_in_cabinet: the cabinet articulation is not
        # in the TRO joints, so it spawns CLOSED. Env vars remain as overrides.
        _declared_doors = tro_state.get("open_doors") or {}
        _door_names = ([n.strip() for n in os.environ.get("OPEN_DOOR_NAMES", "").split(",") if n.strip()]
                       + [str(n) for n in _declared_doors.get("names", [])])
        _door_cats = ([c.strip() for c in os.environ.get("OPEN_DOOR_CATEGORIES", "").split(",") if c.strip()]
                      + [str(c) for c in _declared_doors.get("categories", [])])
        if _door_rooms or _door_names or _door_cats:
            try:
                from omnigibson.object_states import Open as _OpenState
            except Exception:
                _OpenState = None
            _reg = self.env.scene.object_registry
            _doors = set()
            for _rm in _door_rooms:                      # INTERIOR doors connecting the goal room to another room
                for _d in (_reg("in_rooms", _rm) or set()):
                    if "door" not in str(getattr(_d, "category", "")):
                        continue
                    # Only open doors that CONNECT the goal room to another room
                    # (interior path doors, in_rooms >= 2). Skip single-room
                    # boundary/exterior doors — e.g. the garage→outside vehicle
                    # door garage_door_prdjyt_0 (in_rooms=['garage_0']) — which
                    # must stay at its scene default (closed).
                    if len(getattr(_d, "in_rooms", None) or []) >= 2:
                        _doors.add(_d)
            for _cat in _door_cats:                      # explicit category override
                _doors |= (_reg("category", _cat) or set())
            for _nm in _door_names:                      # explicit name override
                _d = _reg("name", _nm)
                if _d is not None:
                    _doors.add(_d)
            # OPEN_DOOR_JOINT_NAMES (comma list, e.g. "j_right_door"): open ONLY
            # those named joints, leaving the rest closed — so e.g. a French-door
            # fridge opens just the right door (matching the demo, which opened a
            # single door) instead of the Open state swinging BOTH.
            _open_joint_names = ([j.strip() for j in os.environ.get("OPEN_DOOR_JOINT_NAMES", "").split(",") if j.strip()]
                                 + [str(j) for j in _declared_doors.get("joint_names", [])])
            # BATCH-CLIENT LEAKAGE GUARD (2026-08-13): the env is REUSED across
            # instances (one boot serves several episodes) and cabinet joints are
            # not in the TRO restore set — doors the previous episode's policy
            # bumped open would persist into this instance. CLOSE every joint of
            # each declared door object first, so instance init is deterministic,
            # then open only the declared joints below.
            for _door in _doors:
                try:
                    _door.set_joint_positions(th.zeros_like(_door.joint_upper_limits), drive=False)
                except Exception as _e:
                    logger.warning(f"Failed to reset joints of {getattr(_door,'name','?')}: {_e}")
            for _door in _doors:
                try:
                    _djoints = getattr(_door, "joints", None) or {}
                    _named = [j for j in _open_joint_names if j in _djoints]
                    if _named:
                        # Open each named joint to its OPEN extreme = the limit
                        # farther from the closed (~0) pose.
                        for _jn in _named:
                            _jl = _djoints[_jn]
                            _lo, _hi = float(_jl.lower_limit), float(_jl.upper_limit)
                            _open_pos = _hi if abs(_hi) >= abs(_lo) else _lo
                            _door.set_joint_positions(th.tensor([_open_pos]), indices=_jl.dof_indices, drive=False)
                        logger.info(f"Opened named joints {_named} of {_door.name} (others left closed)")
                    elif _OpenState is not None and _OpenState in _door.states:
                        _door.states[_OpenState].set_value(True, fully=True)
                        logger.info(f"Opened door (Open state, all) for nav precondition: {_door.name} (category={getattr(_door, 'category', '?')})")
                    else:
                        _door.set_joint_positions(_door.joint_upper_limits, drive=False)
                        logger.info(f"Opened door (all joints) for nav precondition: {_door.name} (category={getattr(_door, 'category', '?')})")
                except Exception as _e:
                    logger.warning(f"Failed to open door {getattr(_door, 'name', '?')}: {_e}")
            # Capture the opened goal-door world xy positions. The nav success
            # criterion (run_single_instance) anchors on reaching the garage
            # DOOR, not on the garage room mask — the mask is the garage interior
            # (it starts ~0.85 m past the door) and its whole entrance edge is
            # wall, so distance-to-room mislabels both door-arrivals and wall-
            # grazes. The door object is the actual gate.
            self._goal_door_xys = []
            for _door in _doors:
                try:
                    _dp = _door.get_position_orientation()[0]
                    _dxy = _dp.cpu().numpy() if hasattr(_dp, "cpu") else np.asarray(_dp)
                    self._goal_door_xys.append([float(_dxy[0]), float(_dxy[1])])
                except Exception as _e:
                    logger.warning(f"[nav] could not read door pos for {getattr(_door, 'name', '?')}: {_e}")
            if self._goal_door_xys:
                logger.info(f"[nav] goal-door anchors (xy): {self._goal_door_xys}")
            if _doors:
                # Re-pin the robot so opening doors doesn't disturb the F0 pose
                # that the stabilize loops just established, then settle.
                self._reassert_robot_joints()
                self.robot.keep_still()
                og.sim.step_physics()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _restore_robot_joints(self, joints: dict) -> None:
        """Set trunk / arm_{L,R} / gripper_{L,R} qpos on the robot.

        `joints` is the torch-converted dict produced by
        ``gen_subtask.robot_proprio_to_joints`` (keys like ``trunk_qpos``,
        ``arm_left_qpos`` …).  Dispatches to ``robot.set_joint_positions``
        per subsystem using the robot's per-subsystem control indices.

        We deliberately DO NOT apply the qvel entries from the TRO: the demo's
        instantaneous joint velocities can be >10 rad/s on an arm joint, and
        applying those before the 25-step physics-only stabilize phase
        launches the arms through the scene.  The 23-dim state the model sees
        only contains base_qvel (proprio[253:256]), not per-joint qvel, so
        starting the arms at rest does not perturb the input distribution.
        We call ``robot.keep_still()`` at the end so stabilize doesn't drift
        the robot under gravity either.
        """
        pos_setters = {
            "trunk_qpos":         self.robot.trunk_control_idx,
            "arm_left_qpos":      self.robot.arm_control_idx["left"],
            "arm_right_qpos":     self.robot.arm_control_idx["right"],
            "gripper_left_qpos":  self.robot.gripper_control_idx["left"],
            "gripper_right_qpos": self.robot.gripper_control_idx["right"],
        }
        applied = []
        self._held_joint_targets = []
        for key, idx in pos_setters.items():
            if key not in joints:
                continue
            # recursively_convert_to_torch doesn't reach leaves inside
            # list-of-dict structures, so qpos may still be a Python list.
            qpos = joints[key]
            if not isinstance(qpos, th.Tensor):
                qpos = th.tensor(qpos, dtype=th.float32)
            self.robot.set_joint_positions(qpos, indices=idx)
            self._held_joint_targets.append((idx, qpos))
            applied.append(key)
        if os.environ.get("DUMP_QPOS"):
            _dd = {}
            for _k in ("trunk_qpos", "arm_left_qpos", "arm_right_qpos"):
                if _k in joints:
                    _v = joints[_k]
                    _v = _v.tolist() if hasattr(_v, "tolist") else list(_v)
                    _dd[_k] = ",".join(f"{float(x):.4f}" for x in _v)
            # robot's symmetric reset config for both arms → derive the L->R mirror.
            for _attr in ("reset_joint_pos", "_reset_joint_pos", "untucked_default_joint_pos"):
                _rj = getattr(self.robot, _attr, None)
                if _rj is None:
                    continue
                try:
                    _rj = _rj.tolist() if hasattr(_rj, "tolist") else list(_rj)
                    for _arm in ("left", "right"):
                        _idx = self.robot.arm_control_idx[_arm]
                        _idx = _idx.tolist() if hasattr(_idx, "tolist") else list(_idx)
                        _dd[f"reset_{_arm}"] = ",".join(f"{float(_rj[int(i)]):.4f}" for i in _idx)
                    _dd["reset_src"] = _attr
                    break
                except Exception as _e:  # noqa: BLE001
                    _dd["reset_err"] = str(_e)[:50]
            print(f"[QPOS_DUMP] {_dd}", flush=True)
        # Optional carry-pose override (place tasks): replace the demo's arm qpos
        # with a fixed "carry" config so a held object sits in the head-cam's
        # work-area view (co-framed with the reference) instead of the demo's
        # raised pose. The matching gripper stays as-restored (closed). The
        # override is appended to _held_joint_targets so _reassert re-pins it each
        # stabilize step (applied last → wins over the demo arm entry). The 7
        # arm qpos values are calibrated via render. Off unless CARRY_ARM_QPOS set.
        _carry = os.environ.get("CARRY_ARM_QPOS", "").strip()
        if _carry:
            try:
                _carm = os.environ.get("CARRY_ARM", "right")
                _cq = th.tensor([float(x) for x in _carry.split(",")], dtype=th.float32)
                _cidx = self.robot.arm_control_idx[_carm]
                self.robot.set_joint_positions(_cq, indices=_cidx)
                self._held_joint_targets.append((_cidx, _cq))
                applied.append(f"carry:{_carm}")
            except Exception as _ce:  # noqa: BLE001
                logger.warning(f"Carry-pose override failed: {_ce}")
        # Optional carry-TRUNK override (place tasks): the arm qpos alone only
        # fixes the eef pose RELATIVE to the trunk tip; the demo's trunk_qpos
        # varies per instance (std ~0.3 rad on carving), so the held object's
        # WORLD height drifts unless the trunk is pinned too. Set CARRY_TRUNK_QPOS
        # to standardize the init height across instances (the "uniform canonical
        # carry pose"). Re-pinned each stabilize step like the arm override. Off
        # unless CARRY_TRUNK_QPOS set.
        _carry_trunk = os.environ.get("CARRY_TRUNK_QPOS", "").strip()
        if _carry_trunk:
            try:
                _tq = th.tensor([float(x) for x in _carry_trunk.split(",")], dtype=th.float32)
                _tidx = self.robot.trunk_control_idx
                self.robot.set_joint_positions(_tq, indices=_tidx)
                self._held_joint_targets.append((_tidx, _tq))
                applied.append("carry:trunk")
            except Exception as _te:  # noqa: BLE001
                logger.warning(f"Carry-trunk override failed: {_te}")
        # Zero joint velocities so the stabilize phase doesn't drift the arm.
        self.robot.keep_still()
        logger.info(f"Restored robot joints from TRO (qpos only): {applied}")

    def _reassert_robot_joints(self) -> None:
        """Re-pin the restored (F0) qpos. Called every stabilize step so an
        unsupported arm doesn't sag under gravity before the policy takes over.
        The aug P0 start backs the robot off the desk, so the F0 arm pose is no
        longer contact-supported; without re-pinning it shakes and the first
        observation goes OOD, corrupting the first AbsoluteAction predictions."""
        for idx, qpos in getattr(self, "_held_joint_targets", []):
            self.robot.set_joint_positions(qpos, indices=idx)


# ── Pre-policy settle: env.step(hold) for N steps so PD controllers actually
# track the F0 target under physics before the policy first speaks (proper
# version of Phase B; the latter only ran og.sim.step_physics + keep_still,
# which doesn't push a fresh command through the controller pipeline).

def _build_hold_action(evaluator):
    """Build a 23-dim BEHAVIOR-layout hold action: zero base velocity +
    current joint qpos as targets.

    Note: OmniGibson's env.action_space.shape is None (not a standard gym Box),
    so we DON'T gate on it. The eval's normal rollout sends 23-dim flat arrays
    via env.step(robot_action) without issue, so we trust the same here."""
    env = evaluator.env
    robot = evaluator.robot
    a = np.zeros(23, dtype=np.float32)
    try:
        qpos = robot.get_joint_positions()
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu().numpy()
        qpos = np.asarray(qpos, dtype=np.float32)
        # 23-dim action layout (matches the state composition):
        #   [0:3]   base_vel       (zero -> hold)
        #   [3:7]   trunk          (4)
        #   [7:14]  arm_left       (7)
        #   [14]    gripper_left   (SUM of both fingers)
        #   [15:22] arm_right      (7)
        #   [22]    gripper_right  (SUM)
        a[3:7]   = qpos[robot.trunk_control_idx][:4]
        a[7:14]  = qpos[robot.arm_control_idx["left"]][:7]
        a[15:22] = qpos[robot.arm_control_idx["right"]][:7]
        gl = qpos[robot.gripper_control_idx["left"]]
        gr = qpos[robot.gripper_control_idx["right"]]
        a[14] = float(np.sum(gl))
        a[22] = float(np.sum(gr))
    except Exception as e:
        logger.warning(f"_build_hold_action: build failed ({e}); skipping settle.")
        return None
    return a


# ── Instruction forwarding to the websocket server ───────────────────────

def _sim_derived_destination_corners(evaluator, goal_conditions) -> "np.ndarray | None":
    """Construct the place-goal 3D bbox corners from LIVE scene geometry.

    Used when the instance cfg carries no `ir_destination_corners_3d` (held-out
    place instances have no source-demo placement). Returns (8,3) world corners
    of a small virtual bbox centred on the constructed goal point, or None when
    the goal conditions aren't a place type / the ref is missing.

    direction conds: goal = ref-AABB centre + (ref half-extent along the axis +
      IR_DEST_DIR_MARGIN) x the ROBOT-relative direction vector (EXACTLY the
      check_direction convention: right = forward rotated -90deg), z = ref top.
    in_layer conds: goal = fridge-AABB footprint centre, z = compartment centre.
    """
    from goal_checker import get_robot_forward_right

    _DIRS = ("left_of", "right_of", "in_front_of", "behind")
    cond = next((c for c in (goal_conditions or [])
                 if c.get("type") in _DIRS + ("in_layer",)), None)
    if cond is None or not cond.get("ref"):
        return None
    try:
        ref = evaluator.env.task.object_scope[cond["ref"]]
    except KeyError:
        logger.warning(f"[with_ir] sim-derived destination: ref {cond['ref']!r} not in scope")
        return None
    try:
        _ru = getattr(ref, "wrapped_obj", ref)
        lo, hi = _ru.aabb
        lo = np.asarray(lo.cpu() if hasattr(lo, "cpu") else lo, dtype=float)
        hi = np.asarray(hi.cpu() if hasattr(hi, "cpu") else hi, dtype=float)
    except Exception as e:
        logger.warning(f"[with_ir] sim-derived destination: ref AABB failed ({e})")
        return None
    ctr = 0.5 * (lo + hi)

    if cond["type"] == "in_layer":
        z_min, z_max = cond.get("z_min"), cond.get("z_max")
        if z_min is None or z_max is None:
            return None
        dest = np.array([ctr[0], ctr[1], 0.5 * (float(z_min) + float(z_max))])
    else:
        margin = float(os.environ.get("IR_DEST_DIR_MARGIN", "0.15"))
        fwd, right = get_robot_forward_right(evaluator.robot)
        axis = {"left_of": -right, "right_of": right,
                "in_front_of": fwd, "behind": -fwd}[cond["type"]]
        ext = hi - lo
        # ref half-extent measured along the (unit, xy) direction axis
        half_ext = 0.5 * (ext[0] * abs(float(axis[0])) + ext[1] * abs(float(axis[1])))
        dxy = ctr[:2] + axis * (half_ext + margin)
        dest = np.array([dxy[0], dxy[1], hi[2]])  # ref top ~ support surface

    h = 0.06  # ~12 cm virtual bbox, roughly target-object scale
    offs = np.array([[sx, sy, sz]
                     for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
    return dest[None, :] + offs


def _send_reset_with_instruction(policy, subtask_config: dict, instance_name: str = "") -> None:
    """Send a `reset` message with an optional `instruction` field to the server.

    The policy server should use the instruction as its prompt from the next
    inference on (docs/protocol.md).

    On the **first** subtask instance of a job, `WebsocketClientPolicy._ws` is
    still `None` (the client only dials out on its first `.reset()` or `.act()`
    call).  We force the connection here so the instruction update lands before
    the evaluator's subsequent plain `reset()` call.  Without this, the server
    would use its CLI `--prompt` for the first instance.
    """
    ws = policy
    # omnigibson.learning.policies.WebsocketPolicy wraps WebsocketClientPolicy
    # under the attribute name `.policy` (see OmniGibson/omnigibson/learning/policies.py).
    if hasattr(ws, "policy") and not hasattr(ws, "_ws"):
        ws = ws.policy
    if hasattr(ws, "_policy") and not hasattr(ws, "_ws"):
        ws = ws._policy
    if hasattr(ws, "model"):
        ws = ws.model

    # Force the connection open if the client hasn't dialled yet.
    if (
        hasattr(ws, "_ws")
        and ws._ws is None
        and hasattr(ws, "_wait_for_server")
    ):
        logger.info("Websocket not yet connected — connecting now before instruction update.")
        ws._ws, ws._server_metadata = ws._wait_for_server()

    if not hasattr(ws, "_ws") or ws._ws is None:
        logger.warning("No active websocket; skipping instruction update.")
        return

    # PROMPT_OVERRIDE (env): replace the instance's referring-expression
    # instruction with a fixed abstract prompt (e.g. "reach the paperback book on
    # the desk"), to match an abstract-prompt-trained policy — so the IR dot, not
    # the language, must identify the target. Unset → use the instance instruction.
    _instruction = os.environ.get("PROMPT_OVERRIDE") or subtask_config.get("instruction", "")
    msg = {
        "reset": True,
        "instruction": _instruction,
        "task_category": subtask_config.get("task_category", ""),
        "task_type": subtask_config.get("task_type", ""),
        "target_instances": subtask_config.get("target_instances", []),
        "instance_name": instance_name,
    }
    ws._ws.send(ws._packer.pack(msg))
    logger.info(
        f"Sent reset+instruction to server: {_instruction!r}"
        + (" (PROMPT_OVERRIDE)" if os.environ.get("PROMPT_OVERRIDE") else "")
    )


# ── Per-instance rollout ─────────────────────────────────────────────────

def _capture_initial_positions(evaluator, goal_conditions: list[dict]) -> dict:
    """Snapshot the 3D positions of all objects referenced by goal conditions.

    Also captures positions of any sibling instances (e.g., paperback_book.n.01_1
    when goal is paperback_book.n.01_2) so goal_checker's diagnostics can compare
    displacements for every instance, helping us identify if the policy grasped
    the wrong instance.
    """
    positions: dict = {}
    scope = evaluator.env.task.object_scope
    # First, record objects explicitly referenced in goal conditions.
    for cond in goal_conditions:
        for key in ("obj", "ref", "target"):
            name = cond.get(key)
            if not name or name in positions or name not in scope:
                continue
            obj = scope[name]
            actual = obj.wrapped_obj if hasattr(obj, "wrapped_obj") else obj
            pos, _ = actual.get_position_orientation()
            positions[name] = (
                pos.cpu().numpy().tolist()
                if hasattr(pos, "cpu") else list(pos)
            )
    # Also snapshot positions for siblings of each goal-referenced object, so
    # goal_checker's diag can tell which instance actually moved.
    import sys
    referenced_prefixes = set()
    for name in positions:
        prefix = name.split(".n.01")[0]  # e.g., "paperback_book"
        referenced_prefixes.add(prefix)
    for k in list(scope.keys()):
        if k in positions:
            continue
        for prefix in referenced_prefixes:
            if prefix in k:
                try:
                    obj = scope[k]
                    actual = obj.wrapped_obj if hasattr(obj, "wrapped_obj") else obj
                    pos, _ = actual.get_position_orientation()
                    positions[k] = (
                        pos.cpu().numpy().tolist()
                        if hasattr(pos, "cpu") else list(pos)
                    )
                except Exception as e:
                    print(f"[init_pos.diag] sibling {k}: error {e}", flush=True)
                break
    # Log captured positions for diagnostics.
    for k, p in positions.items():
        print(f"[init_pos.diag] {k}: {np.round(np.asarray(p), 3).tolist()}", flush=True)
    return positions


def run_single_instance(
    evaluator: SubtaskEvaluator,
    subtask_config_file: str,
    config: DictConfig,
    video_path: Path,
    metrics_path: Path,
    instance_name: str,
) -> bool:
    with open(subtask_config_file) as f:
        subtask_cfg = json.load(f)
    goal_conditions = subtask_cfg.get("goal_conditions", [])
    tro_file = subtask_config_file.replace(".json", "_tro.json")

    logger.info(f"=== Instance: {instance_name} ===")
    logger.info(f"  Instruction: {subtask_cfg.get('instruction', '')}")
    logger.info(f"  TRO: {tro_file}")
    logger.info(f"  Goal conditions: {len(goal_conditions)}")

    # Instruction swap on the server (before loading TRO).
    _send_reset_with_instruction(evaluator.policy, subtask_cfg, instance_name=instance_name)

    # Update the IR target for this subtask (no-op if IR injector not enabled).
    # Source the target object name from goal_conditions[0]["obj"] — that's
    # the OmniGibson scope name (e.g. "paperback_book.n.01_1") that the
    # injector needs for AABB lookup. `subtask_cfg["target_instances"]`
    # doesn't exist in these reach configs; "target_object_id" holds sim
    # ids ("paperback_book_226") which don't match object_scope keys.
    # Passing `instance_name` also names the per-instance IR debug video, if
    # debug_ir_video_dir was set in the cfg.
    ir_injector = getattr(evaluator, "_ir_injector", None)
    if ir_injector is not None:
        ir_targets: list[str] = []
        if goal_conditions:
            primary = goal_conditions[0].get("obj")
            if primary:
                ir_targets.append(primary)
            # TIE SET (2026-08-09): pass the other gt_targets too — the injector
            # dots the first VISIBLE member, so a tie whose primary is off-frame
            # still guides the policy to a correct (visible) tie sibling.
            for gtt in (goal_conditions[0].get("gt_targets") or []):
                if gtt not in ir_targets:
                    ir_targets.append(gtt)
        # Legacy fallback (was target_instances).
        if not ir_targets:
            ir_targets = list(subtask_cfg.get("target_instances", []) or [])
        # DESTINATION mode (place tasks): training IR pins the dot at the GOAL
        # (target bbox frozen at the demo's place-end frame), not on the carried
        # object. Instance cfgs stamped when the eval set was built
        # carry those frozen corners; pass them so eval matches training. Reach
        # cfgs lack the field -> live-object tracking (unchanged).
        ir_dest_corners = subtask_cfg.get("ir_destination_corners_3d")
        if ir_dest_corners is not None:
            logger.info(
                "[with_ir] DESTINATION mode: static goal corners from cfg "
                "(meta=%s)", subtask_cfg.get("ir_destination_meta"),
            )
        # VLM target description: when the POLICY prompt is abstract (ir-abstract
        # policy, PROMPT_OVERRIDE set), the VLM IR predictor still needs the
        # CONCRETE referring expression to point at the object. Ship the
        # instance's real instruction when SHIP_VLM_TARGET_DESC=1 (server feeds
        # it to the VLM). Default off → plain ir path unchanged (VLM uses the
        # policy prompt, which is already concrete there).
        _vlm_desc = None
        if os.environ.get("SHIP_VLM_TARGET_DESC", "0") == "1":
            _vlm_desc = subtask_cfg.get("instruction") or None
        ir_injector.update_target_instances(
            ir_targets, instance_name=instance_name,
            static_world_corners=ir_dest_corners,
            vlm_target_desc=_vlm_desc,
        )

    # Load TRO to set initial state.
    evaluator.reset()
    _door_rooms = [c.get("room_instance") for c in goal_conditions
                   if c.get("type") == "in_room" and c.get("room_instance")]
    # Force-grasp targets = the goal's target object sim_ids (bond ONLY these,
    # not the nearest arbitrary object — avoids grabbing a distractor / missing).
    _grasp_targets = [c.get("obj_sim_id") for c in goal_conditions if c.get("obj_sim_id")]
    evaluator.load_task_instance_from_tro(tro_file, open_doors_for_rooms=_door_rooms,
                                          grasp_target_sim_ids=_grasp_targets)

    initial_positions = _capture_initial_positions(evaluator, goal_conditions)

    # Run the rollout.
    evaluator.reset()

    # Pre-policy settle (env-gated): call env.step(hold_action) for N steps so
    # the robot's PD+gravity-comp controllers actually track the F0 target
    # under physics, reaching a real equilibrium before the policy issues its
    # first action. Inserted BEFORE the video writer + metric callbacks so the
    # settle doesn't show up in metrics, video, or the env step-budget. Hold
    # action = zero base velocity + current joint qpos targets. Default 0 =
    # OFF (no behaviour change); set PRE_POLICY_SETTLE_STEPS=150 for ~5 s at
    # the typical 30 Hz action rate.
    #
    # Bulletproofness (per code-review):
    #   (a) og.sim.app.update() before building the hold action — refreshes
    #       state queries so get_joint_positions() reflects the post-reset
    #       snapshot state, not a stale frame.
    #   (b) robot.apply_action(_hold) BEFORE the first physics step — snaps the
    #       controller setpoints to _hold synchronously so the very first
    #       env.step's physics tick has zero PD tracking error (no jolt).
    #   (c) After the loop, refresh evaluator.obs from the last env.step's
    #       returned obs — otherwise self.obs is the *cached pre-settle*
    #       observation from the 2nd reset() and the policy's first inference
    #       would consume a stale frame while the robot is at the settled state.
    _settle_steps = int(os.environ.get("PRE_POLICY_SETTLE_STEPS", "0"))
    if _settle_steps > 0:
        # (a) refresh state queries
        try:
            og.sim.app.update()
        except Exception:
            pass
        _hold = _build_hold_action(evaluator)
        if _hold is None:
            logger.warning("Pre-policy settle: skipped (hold-action build failed).")
        else:
            # (b) pre-snap controller targets without stepping physics, so the
            # first physics step inside env.step has zero PD tracking error.
            try:
                evaluator.robot.apply_action(_hold)
            except Exception as _e:
                logger.warning(f"Pre-policy settle: apply_action pre-snap failed ({_e}); "
                               f"first env.step's internal apply_action will still snap targets.")
            _settled = 0
            _last_obs = None
            for _i in range(_settle_steps):
                try:
                    _last_obs, _, _term, _trunc, _info = evaluator.env.step(_hold, n_render_iterations=1)
                    _settled += 1
                except Exception as _e:
                    logger.warning(f"Pre-policy settle aborted at step {_i}: {_e}")
                    break
            # (c) refresh evaluator.obs so the policy's first inference sees
            # the post-settle state instead of the cached pre-settle frame.
            if _last_obs is not None:
                try:
                    evaluator.obs = evaluator._preprocess_obs(_last_obs)
                except Exception as _e:
                    logger.warning(f"Pre-policy settle: obs refresh failed ({_e})")
            # Reset env step counter so the rollout's max_steps budget is
            # untouched by the settle phase.
            try:
                evaluator.env._current_step = 0
            except Exception:
                pass
            logger.info(f"Pre-policy settle: {_settled}/{_settle_steps} env.step(hold) done; "
                        f"evaluator.obs refreshed to post-settle frame")

    # ── Sim-derived DESTINATION fallback (place tasks) ───────────────────────
    # Held-out place instances have no demo placement to stamp (novel
    # direction/reference/layer combos), so cfg `ir_destination_corners_3d` is
    # absent and the IR dot would fall back to tracking the CARRIED object —
    # a train/eval mismatch (training marks the GOAL). Construct the goal from
    # live scene geometry instead, AFTER the TRO reset (robot pose is final):
    #   direction: ref AABB edge + IR_DEST_DIR_MARGIN along the ROBOT-relative
    #     direction axis — same forward/right convention as check_direction —
    #     at the ref's top-surface z (margin default 0.15 m: inside the train
    #     demos' 0.10-0.88 m proj distribution).
    #   in_layer:  fridge AABB footprint center at the target compartment's
    #     z center (compartment bands are embedded in every instance).
    # A ~12 cm virtual bbox around the point feeds the injector's existing
    # static-corners DESTINATION mode. Explicit cfg stamps take precedence.
    if ir_injector is not None and ir_dest_corners is None:
        _sim_corners = _sim_derived_destination_corners(evaluator, goal_conditions)
        if _sim_corners is not None:
            ir_injector.update_target_instances(
                ir_targets, instance_name=instance_name,
                static_world_corners=_sim_corners,
            )
            logger.info(
                "[with_ir] DESTINATION mode (sim-derived): goal point %s",
                np.round(_sim_corners.mean(axis=0), 3).tolist(),
            )

    # ── Render warm-up (env-gated, default ON) ──────────────────────────────
    # The first RTX frames after a reset are blurry (denoiser / material
    # warm-up), and the policy computes its first 32-step chunk from that
    # degraded frame — train demos carry no such blur. Render-only iterations
    # (no physics), then refresh evaluator.obs so the first inference sees a
    # converged frame. RENDER_WARMUP_ITERS=0 disables.
    _warmup = int(os.environ.get("RENDER_WARMUP_ITERS", "60"))
    if _warmup > 0:
        for _ in range(_warmup):
            og.sim.render()
        try:
            _wo = evaluator.env.get_obs()
            _wo = _wo[0] if isinstance(_wo, tuple) else _wo
            evaluator.obs = evaluator._preprocess_obs(_wo)
            logger.info(f"Render warm-up: {_warmup} render iters; evaluator.obs refreshed")
        except Exception as _e:
            logger.warning(f"Render warm-up: obs refresh failed ({_e}); "
                           f"first inference may see the pre-warm-up frame")

    # Render quality: OmniGibson's real-time RTX raytracer lags the physics by a
    # few frames and its denoiser must re-converge whenever the state/camera
    # changes, so a single render per frame looks grainy. Warm the renderer once
    # here (on the settled state), then re-render each recorded frame
    # RECORD_RENDER_ITERS extra times in the rollout loop below — OmniGibson's
    # documented remedy. Env-tunable; default-on at modest values.
    _record_render_iters = max(0, int(os.environ.get("RECORD_RENDER_ITERS", "2")))
    if config.write_video:
        for _ in range(max(0, int(os.environ.get("RENDER_WARMUP_STEPS", "8")))):
            og.sim.render()
        video_name = str(video_path / f"{instance_name}.mp4")
        evaluator.video_writer = create_video_writer(
            fpath=video_name, resolution=(448, 672),
        )
    for metric in evaluator.metrics:
        metric.start_callback(evaluator.env)

    # Nav success is tracked per-step by NavSuccessTracker (goal_checker.py):
    # base in a goal room (exact membership) OR within NAV_SUCCESS_DOOR_TOL_M of
    # an opened goal door — anchored on the DOOR, not the room mask. See that
    # class for the rationale and the audit trail it records.
    nav = NavSuccessTracker(
        goal_rooms={c["room_instance"] for c in goal_conditions
                    if c.get("type") == "in_room" and c.get("room_instance")},
        goal_door_xys=getattr(evaluator, "_goal_door_xys", []),
        seg_map=getattr(getattr(evaluator.env, "scene", None), "_seg_map", None),
        door_tol=float(os.environ.get("NAV_SUCCESS_DOOR_TOL_M", "1.3")),
        traj_every=int(os.environ.get("NAV_TRAJ_LOG_EVERY", "5")),
    )
    _seg_map = nav.seg_map  # reused for the final-pose room lookup below

    # Per-step SOFT layer signal for tupperware: did any same-category object ever
    # pass through the band INSIDE the fridge (even if it fell out by the end)?
    _layer_cond = next((c for c in (goal_conditions or [])
                        if c.get("type") == "in_layer"), None)
    layer_tracker = LayerReachTracker(
        evaluator.env, _layer_cond,
        band_tol=float(os.environ.get("LAYER_BAND_TOL", "0.05")),
    )

    # Reach outcome by DWELL, not first touch: the reach BDDL goal terminates
    # the sim on the first touch of ANY candidate, so a transient graze of a
    # distractor en route ends the rollout as 'wrong' even when the arm was
    # heading to the target. With the tracker active, the BDDL termination is
    # ADVISORY: keep stepping until some candidate is touched REACH_DWELL_STEPS
    # consecutive sim steps (that candidate is the outcome) or max-steps
    # truncation. REACH_DWELL_STEPS=0 restores legacy first-touch semantics.
    # K default 30 (~1 s = 3 REPLAN_STEPS cycles; K=10 fit inside ONE open-loop
    # chunk-execution window, so a clutter graze could commit the episode) plus
    # a stationarity guard: the contacting gripper must also be near-still over
    # the window (REACH_DWELL_STATIONARY_EPS metres; 0 disables).
    dwell = ReachDwellTracker(
        evaluator.env, evaluator.robot, goal_conditions,
        dwell_steps=int(os.environ.get("REACH_DWELL_STEPS", "30")),
        stationary_eps=float(os.environ.get("REACH_DWELL_STATIONARY_EPS", "0.03")),
        # REACH_DWELL_TARGET_ONLY=1: non-target dwells do not end the episode;
        # run to truncation and score by closest candidate (user decision
        # 2026-08-08 for candidate-cluttered tasks: soda_can, storage_box).
        target_only=os.environ.get("REACH_DWELL_TARGET_ONLY", "0") == "1",
    )
    if dwell.active:
        logger.info(
            f"[dwell] reach outcome = sustained contact for {dwell.dwell_steps} "
            f"consecutive steps with near-stationary hand "
            f"(eps={dwell.stationary_eps}m; BDDL first-touch termination is advisory)"
        )

    # base_near tasks (soda_can) under REACH_DWELL_TARGET_ONLY=1: the BDDL
    # `(or (touching agent X*))` termination fires on ANY can — including a can
    # the base brushes while WALKING PAST toward the right one — and scoring is
    # base-distance at that frozen moment, so a bypass-graze scores 'wrong'
    # (user issue 2026-08-08). Make that termination ADVISORY exactly like the
    # dwell tracker does for touched_target tasks: continue unless the touched
    # object is a GT target; otherwise run to truncation and let
    # check_base_near_target score the closest can at the end.
    _bn_cond = next((c for c in (goal_conditions or [])
                     if c.get("type") == "base_near_target"), None)
    _bn_advisory = (_bn_cond is not None
                    and os.environ.get("REACH_DWELL_TARGET_ONLY", "0") == "1")
    _bn_gt = set((_bn_cond.get("gt_targets") or [_bn_cond["obj"]])) if _bn_cond else set()
    def _touching_gt() -> bool:
        try:
            from omnigibson.object_states import Touching  # type: ignore
            scope = evaluator.env.task.object_scope
            agent = scope["agent.n.01_1"]
            return any(agent.states[Touching].get_value(scope[g]) for g in _bn_gt if g in scope)
        except Exception:
            return False
    if _bn_advisory:
        logger.info(
            f"[base_near] target-only termination: BDDL touch-stop is advisory "
            f"unless touching gt={sorted(_bn_gt)}; else run to truncation"
        )

    # ── Shadow scoring at program return (E2 agent evals only; OFF unless SHADOW_RETURN_GRACE_STEPS>0) ──
    # Validates ending an agent episode early: SHADOW_RETURN_GRACE_STEPS steps after the agent program
    # returns, score the episode exactly like the end-of-episode code below would at that moment, store it
    # under metrics["shadow_at_return"], and KEEP RUNNING to the normal end. Comparing the two outcomes on
    # the same episode shows whether ending at return+grace would ever change a score. Read-only: it builds
    # its own SubtaskGoalChecker and works on a COPY of the dwell tracker; the real episode is untouched.
    _shadow_grace = int(os.environ.get("SHADOW_RETURN_GRACE_STEPS", "0"))
    _shadow: dict | None = None
    _shadow_ret_step = None

    def _shadow_score() -> dict:
        import copy as _copy
        out: dict = {"return_step": _shadow_ret_step, "scored_step": int(evaluator.env._current_step)}
        if not goal_conditions:
            return out
        chk = SubtaskGoalChecker(env=evaluator.env, robot=evaluator.robot, goal_conditions=goal_conditions,
                                 initial_positions=initial_positions, entered_rooms=nav.entered_rooms)
        succ, res = chk.check_all()
        dw = _copy.copy(dwell)
        if dw.active:
            if getattr(dw, "target_only", False) and not dw.decided:
                dw.decide_at_timeout()
            for k in list(res):
                if k.startswith("touched_target("):
                    res[k] = (dw.outcome == "correct")
            succ = all(res.values())
        out["subtask_success"] = bool(succ)
        td = getattr(chk, "touched_diag", None)
        if td is not None:
            out["touched_target"] = dict(td)
        bd = getattr(chk, "base_near_diag", None)
        if bd is not None:
            out["base_near_target"] = bd
        if dw.active:
            out["touched_target_dwell"] = dw.to_dict()
            if getattr(dw, "target_only", False) and "touched_target" in out:
                out["touched_target"]["outcome"] = dw.outcome
                if getattr(dw, "closest_at_timeout", None) is not None:
                    out["touched_target"]["closest_at_timeout"] = dw.closest_at_timeout
        ld = getattr(chk, "layer_diag", None)
        if ld is not None:
            out["in_layer"] = ld
        dd = getattr(chk, "direction_diag", None)
        if dd is not None:
            out["direction"] = dd
        # final_state geometry exactly as the end-of-episode block records it (drives soft+/soft-)
        try:
            from goal_checker import get_object_pos as _gpos2
            tref = subtask_cfg.get("target_bddl_ref")
            crefs = list(subtask_cfg.get("candidate_bddl_refs") or [])
            if tref and tref not in crefs:
                crefs.append(tref)
            scope = evaluator.env.task.object_scope
            eefs = {}
            for arm in evaluator.robot.arm_names:
                pp = evaluator.robot.get_eef_position(arm)
                pp = pp.cpu().numpy() if hasattr(pp, "cpu") else pp
                eefs[arm] = np.asarray(pp, dtype=float).tolist()
            objs = {}
            for ref in crefs:
                if ref and ref in scope:
                    o = scope[ref]
                    o = o.wrapped_obj if hasattr(o, "wrapped_obj") else o
                    objs[ref] = _gpos2(o).tolist()
            dists = {}
            for ref, op in objs.items():
                per = {a: float(np.linalg.norm(np.asarray(e) - np.asarray(op, dtype=float))) for a, e in eefs.items()}
                per["min"] = min(per[a] for a in eefs)
                dists[ref] = per
            fs = {"eef_positions": eefs, "object_positions": objs, "distances_to_candidates": dists,
                  "target_bddl_ref": tref}
            if tref and tref in dists:
                fs["distance_to_target"] = dists[tref]
            out["final_state"] = fs
        except Exception as _e:
            out["final_state_error"] = repr(_e)
        return out

    # Start state for the soft lean (summarize_eval_results.py): the distances the
    # end-of-episode checks record, taken before the first policy action.
    try:
        _start_state = SubtaskGoalChecker(env=evaluator.env, robot=evaluator.robot,
                                          goal_conditions=goal_conditions).start_state()
    except Exception as _e:  # never let the snapshot affect the episode
        _start_state = {"errors": [repr(_e)]}

    terminated = truncated = False
    done = False
    while not done:
        terminated, truncated = evaluator.step()
        if _bn_advisory:
            if truncated or (terminated and _touching_gt()):
                done = True
        elif dwell.active:
            try:
                dwell.update(evaluator.env._current_step)
            except Exception:
                pass
            # first-touch `terminated` is advisory here; stop on dwell/timeout.
            if truncated or dwell.decided:
                done = True
        elif terminated or truncated:
            done = True
        if nav.active:
            try:
                nav.update(evaluator.robot.get_position_orientation()[0][:2],
                           evaluator.env._current_step)
            except Exception:
                pass
        if layer_tracker.active:
            try:
                layer_tracker.update(evaluator.env._current_step)
            except Exception:
                pass
        if _shadow_grace > 0 and _shadow is None:
            try:
                if _shadow_ret_step is None and getattr(getattr(evaluator, "policy", None), "agent_finished", False):
                    _shadow_ret_step = int(evaluator.env._current_step)
                if _shadow_ret_step is not None and evaluator.env._current_step - _shadow_ret_step >= _shadow_grace:
                    _shadow = _shadow_score()
                    logger.info(f"[shadow] scored at step {evaluator.env._current_step} "
                                f"(program returned at {_shadow_ret_step})")
            except Exception as _e:  # never let the shadow affect the real episode
                _shadow = {"error": repr(_e)}
        if config.write_video:
            for _ in range(_record_render_iters):
                og.sim.render()
            evaluator._write_video()
        if evaluator.env._current_step % 500 == 0:
            logger.info(f"  Step: {evaluator.env._current_step}")

    for metric in evaluator.metrics:
        metric.end_callback(evaluator.env)
    logger.info(
        f"Finished at step {evaluator.env._current_step} "
        f"(terminated={terminated}, truncated={truncated}); "
        f"BDDL trials={evaluator.n_trials} successes={evaluator.n_success_trials}"
    )
    # Final base pose + room — so we can see WHERE the robot ended vs the goal
    # room (e.g. stuck at the boundary). Logged to .err AND saved into the
    # metrics json (the "nav" block below) for analysis.
    _final_room = None
    _final_base_xyz = None
    # Nav-only: the final-room + door audit log is for move_to tasks. Manipulation
    # tasks (no in_room goal → nav.active is False) skip this block entirely.
    if nav.active and _seg_map is not None:
        try:
            _fp = evaluator.robot.get_position_orientation()[0]
            _froom = _seg_map.get_room_instance_by_point(_fp[:2])
            _final_room = str(_froom) if _froom is not None else None
            _final_base_xyz = [round(float(v), 3) for v in _fp[:3]]
            logger.info(
                f"[nav] FINAL base xyz={_final_base_xyz} "
                f"final_room={_froom}  reached_goal_rooms={sorted(nav.entered_rooms)} "
                f"(door_tol={nav.door_tol}m min_door_dist={nav.min_door_dist:.2f}m "
                f"reached_step={nav.reached_step}@{nav.reached_xy} goal_rooms={sorted(nav.goal_rooms)})"
            )
        except Exception as _e:
            logger.warning(f"[nav] final-pose log failed: {_e}")

    # Flush any pending IR debug video for this instance.
    if ir_injector is not None:
        ir_injector.flush_video()

    # Custom spatial goal check.
    subtask_success = False
    subtask_results: dict = {}
    if goal_conditions:
        checker = SubtaskGoalChecker(
            env=evaluator.env,
            robot=evaluator.robot,
            goal_conditions=goal_conditions,
            initial_positions=initial_positions,
            entered_rooms=nav.entered_rooms,
        )
        subtask_success, subtask_results = checker.check_all()
        # Dwell override: the reach outcome is the candidate the hand STAYED on
        # (ReachDwellTracker), not the instantaneous end-of-rollout contact.
        # check_touched_target's touched_diag is kept in the metrics as the
        # instantaneous view for comparison.
        if dwell.active:
            # target_only mode + no GT dwell by truncation -> closest-candidate
            # decision (fills dwell.outcome/closest_at_timeout).
            if getattr(dwell, "target_only", False) and not dwell.decided:
                dwell.decide_at_timeout()
            for k in list(subtask_results):
                if k.startswith("touched_target("):
                    subtask_results[k] = (dwell.outcome == "correct")
            subtask_success = all(subtask_results.values())
            logger.info(
                f"[dwell] outcome={dwell.outcome} decided_step={dwell.decided_step} "
                f"candidates={dwell.decided_candidates} -> subtask_success={subtask_success}"
            )

    metrics: dict = {}
    for metric in evaluator.metrics:
        metrics.update(metric.gather_results())
    # Surface analysis-relevant identifiers from the subtask config so a
    # downstream summarizer can group metrics by (slice, variant) without
    # needing to load the original config file again.
    metrics.update({
        "subtask_success": bool(subtask_success),
        "subtask_goal_results": {k: bool(v) for k, v in subtask_results.items()},
        "instruction": subtask_cfg.get("instruction", ""),
        "skill_description": subtask_cfg.get("skill_description", ""),
        "config_file": subtask_config_file,
        "slice": subtask_cfg.get("slice"),
        "variant_id": subtask_cfg.get("variant_id"),
        "demo_id": subtask_cfg.get("demo_id"),
        "test_task_name": subtask_cfg.get("test_task_name"),
        "task_type": subtask_cfg.get("task_type"),
        "split_role": subtask_cfg.get("split_role"),
    })
    # Nav outcome + door audit + downsampled trajectory — enough to AUDIT every
    # success/fail and to re-score offline under a different criterion without
    # re-running. Only for nav tasks (an in_room goal is present).
    if nav.active:
        metrics["nav"] = nav.to_metrics(_final_room, _final_base_xyz)
    # Surface the touched_target diagnostic (which candidate the agent was in
    # contact with at end-of-rollout, plus correct/wrong/none label) so reach
    # eval analysis can split outcomes per (slice, variant).
    if goal_conditions:
        touched_diag = getattr(checker, "touched_diag", None)
        if touched_diag is not None:
            metrics["touched_target"] = touched_diag
        # base_near_target (floor-scatter reach, e.g. soda_can): base-position
        # outcome diag — same correct/wrong/none contract as touched_target.
        base_near_diag = getattr(checker, "base_near_diag", None)
        if base_near_diag is not None:
            metrics["base_near_target"] = base_near_diag
        # Dwell audit block (which candidate the hand STAYED on, streaks, first-
        # touch steps) — the authoritative reach outcome when active; the
        # touched_target block above stays as the instantaneous view.
        if dwell.active:
            metrics["touched_target_dwell"] = dwell.to_dict()
            # In target_only mode the summarizer-facing outcome MUST be the
            # dwell/timeout decision, not the instantaneous end-of-rollout
            # contact: at truncation the hand usually touches nothing, which
            # would mis-score a closest-candidate decision as 'none'. The
            # instantaneous label is preserved in outcome_instantaneous.
            if getattr(dwell, "target_only", False) and "touched_target" in metrics:
                metrics["touched_target"]["outcome_instantaneous"] = metrics["touched_target"].get("outcome")
                metrics["touched_target"]["outcome"] = dwell.outcome
                if getattr(dwell, "closest_at_timeout", None) is not None:
                    metrics["touched_target"]["closest_at_timeout"] = dwell.closest_at_timeout
                    metrics["touched_target"]["timeout_dists"] = dwell.timeout_dists
        # Locate-task (tupperware) layer outcome: correct / in_fridge_wrong /
        # not_in_fridge + obj_z, for the per-layer breakdown + closest-layer split.
        layer_diag = getattr(checker, "layer_diag", None)
        if layer_diag is not None:
            metrics["in_layer"] = layer_diag
        if layer_tracker.active:
            metrics["layer_reached"] = layer_tracker.to_dict()
        direction_diag = getattr(checker, "direction_diag", None)
        if direction_diag is not None:
            metrics["direction"] = direction_diag

    # Final-state geometry: EEF positions, target/candidate positions, and
    # per-EEF distances to each candidate. Lets downstream analysis report
    # avg min-EEF-to-target distance per (slice, variant), and tell whether
    # a "wrong" outcome had the EEF nonetheless very close to the GT (= near
    # miss) vs grossly far off.
    target_bddl_ref = subtask_cfg.get("target_bddl_ref")
    candidate_bddl_refs = list(subtask_cfg.get("candidate_bddl_refs") or [])
    if target_bddl_ref and target_bddl_ref not in candidate_bddl_refs:
        candidate_bddl_refs.append(target_bddl_ref)
    try:
        from goal_checker import get_object_pos as _gpos
        obj_scope = evaluator.env.task.object_scope

        eef_positions: dict = {}
        for arm in evaluator.robot.arm_names:
            p = evaluator.robot.get_eef_position(arm)
            if hasattr(p, "cpu"):
                p = p.cpu().numpy()
            eef_positions[arm] = np.asarray(p, dtype=float).tolist()

        object_positions: dict = {}
        for ref in candidate_bddl_refs:
            if not ref or ref not in obj_scope:
                continue
            o = obj_scope[ref]
            o = o.wrapped_obj if hasattr(o, "wrapped_obj") else o
            object_positions[ref] = _gpos(o).tolist()

        distances_to_candidates: dict = {}
        for ref, op in object_positions.items():
            op_arr = np.asarray(op, dtype=float)
            per_arm = {}
            for arm, eef_p in eef_positions.items():
                per_arm[arm] = float(np.linalg.norm(np.asarray(eef_p) - op_arr))
            per_arm["min"] = min(per_arm[a] for a in eef_positions.keys())
            distances_to_candidates[ref] = per_arm

        final_state = {
            "eef_positions": eef_positions,
            "object_positions": object_positions,
            "distances_to_candidates": distances_to_candidates,
            "target_bddl_ref": target_bddl_ref,
        }
        if target_bddl_ref and target_bddl_ref in distances_to_candidates:
            final_state["distance_to_target"] = distances_to_candidates[target_bddl_ref]
        metrics["final_state"] = final_state
    except Exception as e:
        logger.warning(f"Failed to record final_state: {e}")

    if _shadow_grace > 0:
        # None => the episode ended before return+grace, i.e. ending early would not have changed anything.
        metrics["shadow_at_return"] = {"grace_steps": _shadow_grace, "return_step": _shadow_ret_step, "score": _shadow}
    metrics["start_state"] = _start_state
    with open(metrics_path / f"{instance_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Subtask success: {subtask_success}")

    if config.write_video:
        evaluator.video_writer = None
        logger.info(f"Saved video: {video_name}")

    return subtask_success


# ── Main (Hydra entry point) ─────────────────────────────────────────────

def main() -> None:
    configs_dir = str(_THIS_DIR / "configs")
    with hydra.initialize_config_dir(configs_dir, version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)

    gm.HEADLESS = config.headless

    log_path = Path(config.log_path).expanduser()
    video_path = log_path / "videos"
    metrics_path = log_path / "metrics"
    video_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    # +subtask.config_file accepts a single file or a comma-separated list.
    cfg_str = OmegaConf.select(config, "subtask.config_file", default=None)
    if not cfg_str:
        logger.error("Provide at least one subtask config with +subtask.config_file=...")
        sys.exit(1)
    subtask_cfg_files = [s.strip() for s in cfg_str.split(",") if s.strip()]
    logger.info(f"Will run {len(subtask_cfg_files)} subtask instance(s)")

    # Group configs by task_name so each group shares one env.
    task_groups: OrderedDict[str, list[str]] = OrderedDict()
    for f_path in subtask_cfg_files:
        with open(f_path) as f:
            sc = json.load(f)
        task_groups.setdefault(sc["task_name"], []).append(f_path)

    # IsaacSim allows only ONE simulation app per process, so running multiple OG
    # task groups in THIS process would die after the first env closes. The
    # launcher (run_eval.py) splits the config list by task_name and invokes
    # this script ONCE PER GROUP (separate processes), so we normally see exactly
    # one group here. If invoked directly with a multi-task list, warn — only the
    # first group would actually run.
    if len(task_groups) > 1:
        logger.warning(
            f"{len(task_groups)} OG task groups in one process; IsaacSim can run "
            f"only the first ('{next(iter(task_groups))}'). Launch via "
            f"run_eval.py, which runs each task_name in its own process."
        )

    n_total = len(subtask_cfg_files)
    n_pass = 0

    for task_name, group in task_groups.items():
        logger.info(f"\n{'=' * 60}\n  Task group: {task_name} ({len(group)})\n{'=' * 60}")
        group_cfg = deepcopy(config)
        group_cfg.task.name = task_name

        # Pick the first subtask's BDDL as the group's problem (they share a task).
        bddl_file = OmegaConf.select(group_cfg, "subtask.bddl_file", default=None)
        if bddl_file is None:
            bddl_file = group[0].replace(".json", ".bddl")
        with open(bddl_file) as f:
            bddl_string = f.read()
        logger.info(f"BDDL: {bddl_file}")

        with SubtaskEvaluator(group_cfg, bddl_string=bddl_string) as evaluator:
            # Optionally wrap the policy to inject object-center IR pixel
            # coords into every obs dict — required for the `*_with_ir`
            # policies trained with a target-center input.
            # The server renders its IR image from these coords;
            # see privileged_ir/object_center_ir_omnigibson.py for the math
            # (as the training data was rendered).
            evaluator._ir_injector = None  # type: ignore[attr-defined]
            if OmegaConf.select(group_cfg, "with_ir", default=False):
                ir_video_dir_cfg = OmegaConf.select(
                    group_cfg, "debug_ir_video_dir", default=None,
                )
                ir_video_dir = (
                    Path(ir_video_dir_cfg) if ir_video_dir_cfg else None
                )
                evaluator._ir_injector = IRObsInjector(  # type: ignore[attr-defined]
                    evaluator.robot, evaluator.env.task.object_scope,
                    debug_video_dir=ir_video_dir,
                    with_first_frame=OmegaConf.select(
                        group_cfg, "with_first_frame", default=False,
                    ),
                )
                wrap_policy_forward(evaluator.policy, evaluator._ir_injector)
                logger.info(
                    "[with_ir] IR obs injector wrapped policy.forward "
                    "(debug video dir=%s)", ir_video_dir,
                )
            for i, cfg_file in enumerate(group):
                instance_name = Path(cfg_file).stem
                logger.info(f"\n--- Instance {i + 1}/{n_total}: {instance_name}")
                try:
                    ok = run_single_instance(
                        evaluator, cfg_file, group_cfg,
                        video_path, metrics_path, instance_name,
                    )
                    if ok:
                        n_pass += 1
                except Exception:
                    import traceback
                    logger.error(f"Instance {instance_name} crashed:")
                    traceback.print_exc()

    logger.info(f"\n{'=' * 60}\n  DONE: {n_pass}/{n_total} passed\n{'=' * 60}")


if __name__ == "__main__":
    main()
