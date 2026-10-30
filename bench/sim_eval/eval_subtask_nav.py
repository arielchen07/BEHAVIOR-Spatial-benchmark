"""Evaluate a policy on the navigation (move_to) subtasks.

Separate from eval_subtask.py because the manipulation script's stabilize /
pin-hold loop teleports the gripper every physics step, which ejects carried
objects during navigation. Supports both sim stacks: OmniGibson 3.7.2 (2025
challenge scenes) and OmniGibson 3.9.2 / Isaac Sim 5.1 (2026 restaurant_diner
scene). Use bench/sim_eval/run_eval.py, which selects this script for
instances whose config has task_type == "nav".
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


def _disable_steam_fire_emitters() -> None:
    """restaurant_diner (BEHAVIOR-1K 2026) SEGFAULTS on scene load without this.

    `bottom_cabinet_olgoza_0` carries a non-uniform scale [2.11, 2.30, 1.68] plus
    a rotation. StatefulObject._initialize -> _create_emitter_apis(STEAM) builds
    an `emitter` child prim, and xform_prim.convert_world_pose_to_local asserts
    the local transform is orthogonal — which a rotated non-uniform scale never
    is. The assert fires inside a physics callback and takes the process down:

      stateful_object.py:171 _create_emitter_apis(EmitterType.STEAM)
      -> usd_utils.py:712 AssertionError: .../base_link/emitter local transform
         is not orthogonal   -> Fatal Python error: Segmentation fault

    The emitter is a purely cosmetic steam/fire visual (Heated / OnFire), never
    read by nav evaluation, so it is no-op'd rather than disabling object states
    wholesale — `Open` must keep working for the door-opening path. Idempotent
    and harmless for every other scene. Mirrors the same fix in
    the diner context-tour collection script.
    """
    # 3.7.x defines _create_emitter_apis on StatefulObject; 3.9.x moved it to
    # USDObject. Patch whichever class actually owns it.
    # Import each candidate SEPARATELY: 3.9.x deleted objects/stateful_object.py
    # (only dataset/light/primitive/usd_object remain), so a combined import fails
    # outright and silently skips the patch on BOTH classes.
    targets = []
    for _mod, _cls in (("omnigibson.objects.stateful_object", "StatefulObject"),
                       ("omnigibson.objects.usd_object", "USDObject")):
        try:
            _m = __import__(_mod, fromlist=[_cls])
            _c = getattr(_m, _cls)
            if "_create_emitter_apis" in vars(_c):
                targets.append(_c)
        except (ImportError, AttributeError):
            continue
    try:
        if not targets:
            logger.warning("[scene] no _create_emitter_apis found; emitter patch skipped")
            return
        for cls in targets:
            if getattr(cls, "_emitters_disabled", False):
                continue
            cls._create_emitter_apis = lambda self, emitter_type: None
            cls._emitters_disabled = True
        logger.info("[scene] steam/fire emitter APIs disabled (restaurant_diner load fix)")
    except Exception as _e:  # noqa: BLE001
        logger.warning(f"[scene] could not disable emitter APIs: {_e}")
from omnigibson.macros import gm  # noqa: E402
# OmniGibson >=3.9.0 renamed `omnigibson.learning` -> `omnigibson.eval` and split
# Evaluator out into its own module. Import from whichever tree is installed so this
# script runs unchanged against 3.7.2 (2025 tasks) and 3.9.2 (2026 challenge tasks).
try:  # OmniGibson >= 3.9
    from omnigibson.eval.evaluator import Evaluator  # noqa: E402
    from omnigibson.eval.utils.eval_utils import (  # noqa: E402
        PROPRIOCEPTION_INDICES,
        TASK_NAMES_TO_INDICES,
        generate_basic_environment_config,
    )
    from omnigibson.eval.utils.obs_utils import create_video_writer  # noqa: E402
    _OG_EVAL_NS = "omnigibson.eval"
except ImportError:  # OmniGibson 3.7.x
    from omnigibson.learning.eval import Evaluator  # noqa: E402
    from omnigibson.learning.utils.eval_utils import (  # noqa: E402
        PROPRIOCEPTION_INDICES,
        TASK_NAMES_TO_INDICES,
        generate_basic_environment_config,
    )
    from omnigibson.learning.utils.obs_utils import create_video_writer  # noqa: E402
    _OG_EVAL_NS = "omnigibson.learning"

# 3.9.x dropped the ROBOT_CAMERA_NAMES constant in favour of
# get_robot_camera_names(robot_name, robot_eval_config), which reads
# omnigibson/eval/r1pro.yaml.  That yaml lists exactly the same three R1Pro
# sensors under the same `robot_r1` namespace, so the two agree byte for byte;
# keep the constant so call sites are version-independent.
try:
    from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES  # noqa: E402
except ImportError:
    ROBOT_CAMERA_NAMES = {
        "R1Pro": {
            "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
            "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
            "head": "robot_r1::robot_r1:zed_link:Camera:0",
        },
    }
from omnigibson.utils.python_utils import recursively_convert_to_torch  # noqa: E402

# gello moved these out of robots/sim_robot/ into utils/ for the 2026 release.
try:  # gello shipped with OmniGibson >= 3.9
    from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES  # noqa: E402
    from gello.utils.og_teleop_utils import (  # noqa: E402
        augment_rooms,
        generate_robot_config,
        get_task_relevant_room_types,
        load_available_tasks,
    )
except ImportError:  # gello shipped with OmniGibson 3.7.x
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
from goal_checker import SubtaskGoalChecker, NavSuccessTracker, ObjectNavSuccessTracker  # noqa: E402,I001


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

_EXTERIOR_ROOM_PREFIXES = (
    "garden", "lawn", "outdoor", "backyard", "yard", "porch", "patio", "driveway", "balcony",
)


def _is_exterior_room(room_name) -> bool:
    """True if the room is an outdoor/exterior area (garden, lawn, ...). Doors onto
    these (e.g. the kitchen→garden door) must stay at their scene default (closed)
    during nav — opening them exposes the robot to OOD outdoor space."""
    rn = str(room_name).lower()
    return any(rn.startswith(p) for p in _EXTERIOR_ROOM_PREFIXES)


class _NullPolicy:
    """No-op stand-in for the websocket policy in SPAWN_VALIDATE mode, so spawn
    validation needs no running policy server (no GPU model load / coordination)."""

    def reset(self, *a, **k) -> None:
        return None

    def forward(self, *a, **k):
        return {}


# ── 2026 task registration ───────────────────────────────────────────────────
# The installed OmniGibson/joylo checkout predates the 2026 challenge release, so
# BOTH task registries it consults are 2025-only:
#   * eval_utils.TASK_NAMES_TO_INDICES        -- 47 entries
#   * joylo sampled_task/available_tasks.yaml -- 74 entries (Oct 2025)
# Neither knows the restaurant_diner activities, so load_env() raised
#   ValueError: Unknown task.name: putting_dirty_dishes_in_sink
# We inject them here rather than editing the shared package/yaml, so the change
# is local to this eval script and trivially reversible.
#
# Indices are the authoritative 2026 ones from
# the BEHAVIOR 2026 challenge demos' meta/tasks.parquet.
# robot_start_* are the demos' own initial poses; they are only a fallback --
# every generated instance overrides them via the TRO's robot_poses.
_TASKS_2026 = {
    "putting_dirty_dishes_in_sink": (85, [-4.06, -5.57, 0.0043], [0.0, 0.0, 0.70, 0.71]),
    "cook_a_frozen_pie":            (72, [-8.45, -2.16, 0.0043], [0.0, 0.0, 0.97, 0.23]),
    "cook_brussels_sprouts":        (95, [-7.60, -5.60, 0.0043], [0.0, 0.0, 0.70, 0.71]),
    "clean_up_broken_glass":        (80, [-4.10, -5.60, 0.0043], [0.0, 0.0, 0.70, 0.71]),
}


def _register_2026_scene_robot_class() -> None:
    """Let the scene loader SKIP the robot embedded in 2026 task-instance files.

    scene_base._should_load_object decides by CLASS NAME:
        return self._include_robots or obj_info["class_name"] not in REGISTERED_ROBOTS
    The 2026 templates carry one robot entry authored against a newer OmniGibson:
        robot_slwzxc -> omnigibson.robots.robot / class "Robot"
    This build has no such module or class, so "Robot" is not in REGISTERED_ROBOTS,
    the entry is NOT recognised as a robot, is therefore NOT skipped despite
    include_robots=False, and create_object_from_init_info() dies with
        ModuleNotFoundError: No module named 'omnigibson.robots.robot'
    Registering the name (value is irrelevant -- only membership is tested, and
    include_robots is False for eval) makes the skip fire. The env still builds
    the real R1Pro from its own `robots:` config, as it does for 2025 scenes.
    """
    # STILL REQUIRED on 3.9.x, for a NEW reason. scene_base._should_load_object is
    # byte-identical across versions:
    #     return self._include_robots or obj_info["class_name"] not in REGISTERED_ROBOTS
    # but 3.9.x re-keyed REGISTERED_ROBOTS from CLASS NAMES ("R1Pro", ...) to MODEL
    # IDS ("r1", "r1pro", ...). The 2026 templates' embedded robot has
    # class_name "Robot", which matches no model id, so the guard evaluates True and
    # the robot is loaded EVEN THOUGH include_robots=False. env_base then skips
    # building our configured robot (`if len(self.scene.robots) == 0`), leaving a
    # camera-less "robot_slwzxc" in the scene and failing eval-camera validation.
    # Registering the name restores the skip (only membership is tested).
    try:  # 3.7.x
        from omnigibson.robots.robot_base import REGISTERED_ROBOTS, BaseRobot
        _placeholder = BaseRobot
    except ImportError:  # 3.9.x — robot_base is gone; one unified Robot class
        from omnigibson.robots import REGISTERED_ROBOTS
        from omnigibson.robots.robot import Robot as _placeholder
    # 3.7.x REGISTERED_ROBOTS is a dict {class_name: cls}; 3.9.x made it a LIST of
    # model ids. Only membership is ever tested, so add the name either way.
    if "Robot" in REGISTERED_ROBOTS:
        return
    if hasattr(REGISTERED_ROBOTS, "setdefault"):
        REGISTERED_ROBOTS.setdefault("Robot", _placeholder)
    else:
        REGISTERED_ROBOTS.append("Robot")
    logger.info("[scene] registered 'Robot' so include_robots=False skips the "
                "template's embedded robot")


def _register_2026_tasks(available_tasks: dict) -> None:
    for name, (idx, pos, ori) in _TASKS_2026.items():
        TASK_NAMES_TO_INDICES.setdefault(name, idx)
        if name not in available_tasks:
            available_tasks[name] = {0: {
                "scene_model": "restaurant_diner",
                "robot_start_position": pos,
                "robot_start_orientation": ori,
            }}

def _metric_start(metrics, env) -> None:
    """Begin per-episode metric tracking.

    OmniGibson 3.9.x replaced the MetricBase callback trio
    (start_callback / end_callback / gather_results) with
    reset() / step() / aggregate(env). Dispatch on what the installed class
    actually provides so the harness runs on either tree.
    """
    for m in metrics:
        (m.reset if hasattr(m, "reset") else m.start_callback)(env)


def _metric_end(metrics, env) -> None:
    """End-of-episode hook. 3.9.x has no equivalent -- aggregate() does the work."""
    for m in metrics:
        if hasattr(m, "end_callback"):
            m.end_callback(env)


def _metric_results(metrics, env) -> dict:
    """Collect metric values. 3.7.x: gather_results(); 3.9.x: aggregate(env)."""
    out: dict = {}
    for m in metrics:
        vals = m.gather_results() if hasattr(m, "gather_results") else m.aggregate(env)
        out.update(vals or {})
    return out


def _maybe_disable_ws_keepalive() -> None:
    """WS_DISABLE_KEEPALIVE=1 -> connect websockets with ping_interval=None.

    HYPOTHESIS, not established fact. Eval clients intermittently die with
    `rc=134`: a Carbonite assertion `m_owner == pthread_self()` ("unlock() called
    by non-owning thread") raised from Kit's mutex, mid-episode. Two hints point
    at the websocket keepalive thread:
      * it appears in the fatal-error thread dump (weak: Python dumps EVERY
        thread on abort, so this alone proves nothing), and
      * failures scale with EPISODE LENGTH -- move_to_corner_booth (max_steps
        4000) aborts far more often than oven (2500), across both checkpoints
        and both slices.
    omnigibson/eval/utils/network_utils.py connects with ping_interval=60, so a
    background thread wakes every 60 s alongside Isaac. Disabling it removes that
    concurrency. Safe here: client/server pairs are short-lived and same-cluster,
    and the episode has its own step budget and CLIENT_TIME, so we do not rely on
    keepalive to notice a dead peer.
    """
    if os.environ.get("WS_DISABLE_KEEPALIVE", "0") != "1":
        return
    try:
        import websockets.sync.client as _wsc
        if getattr(_wsc, "_keepalive_disabled", False):
            return
        _orig = _wsc.connect

        def _connect(*a, **kw):
            kw["ping_interval"] = None
            return _orig(*a, **kw)

        _wsc.connect = _connect
        _wsc._keepalive_disabled = True
        logger.info("[ws] keepalive DISABLED (ping_interval=None) — rc=134 abort probe")
    except Exception as _e:  # noqa: BLE001
        logger.warning(f"[ws] could not disable keepalive: {_e}")


# Articulated objects that must be OPEN for a scene to be traversable / to match
# the context-frame bank, but which are NOT goal doors and must never become nav
# success anchors.
#
# restaurant_diner: `bar_xxftww_0` is the bar's flip-up pass-through section (BDDL
# calls it `flipup_countertop`; there is no separate object by that name). The
# context tour drives it open (the diner context-tour collection script
# --open-objects), so leaving it shut at eval both blocks the staff route between
# the dining room and the service side and makes the scene disagree with the
# context images the policy localises against. The kitchen door itself is handled
# per-instance through the generator's `doors_open`.
_SCENE_ALWAYS_OPEN = {
    "restaurant_diner": ["bar_xxftww_0"],
}


def _robot_model_key(robot, tro_data: dict) -> str:
    """Resolve the key a TRO uses for the robot (e.g. "R1Pro").

    3.7.x exposes ``robot.model_name`` -> "R1Pro", matching how the TRO files are
    keyed. 3.9.x rewrote omnigibson/robots/ and dropped ``model_name`` entirely;
    only ``robot.model`` survives, and ``Evaluator._build_robot_config``
    lowercases it to "r1pro" -- so a straight lookup KeyErrors on every instance.
    Match the TRO's own keys case-insensitively, and fall back to the sole entry
    when there is exactly one robot, so this survives further renames.
    """
    cand = getattr(robot, "model_name", None) or getattr(robot, "model", None)
    if cand in tro_data:
        return cand
    if cand:
        lowered = {k.lower(): k for k in tro_data}
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    if len(tro_data) == 1:
        return next(iter(tro_data))
    raise KeyError(f"robot model {cand!r} not found in TRO keys {list(tro_data)}")


class SubtaskEvaluator(Evaluator):
    """Extends OmniGibson's Evaluator to inject a custom BDDL + TRO."""

    def __init__(self, cfg: DictConfig, bddl_string: str | None = None):
        self._bddl_string = bddl_string
        super().__init__(cfg)

    def load_policy(self):
        _maybe_disable_ws_keepalive()
        # Spawn validation only inspects physics; skip the websocket policy so no
        # server is required (see _validate_spawn / SPAWN_VALIDATE).
        if os.environ.get("SPAWN_VALIDATE") == "1":
            logger.info("[spawnval] SPAWN_VALIDATE=1 — using _NullPolicy (no server).")
            return _NullPolicy()
        return super().load_policy()

    def _write_video(self) -> None:
        """Head-only video: this nav policy's visual input is the HEAD camera only
        (model.image_keys=('base_0_rgb',)); the wrist views are NOT fed to the
        policy, so the parent's head+wrist composite is misleading. Render just the
        head (what the model actually sees). Set VIDEO_INCLUDE_WRISTS=1 to restore
        the full head+wrist composite."""
        if os.environ.get("VIDEO_INCLUDE_WRISTS", "0") == "1":
            return super()._write_video()
        import cv2
        if _OG_EVAL_NS == "omnigibson.eval":
            from omnigibson.eval.utils.obs_utils import write_video
        else:
            from omnigibson.learning.utils.obs_utils import write_video
        head_key = ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"
        if head_key not in self.obs:
            return
        head_rgb = cv2.resize(self.obs[head_key].numpy(), (448, 448))
        write_video(
            np.expand_dims(head_rgb, 0),
            video_writer=self.video_writer, batch_size=1, mode="rgb",
        )

    def load_env(self, env_wrapper: DictConfig):
        """Same as parent but pushes our BDDL string into cfg['task']['predefined_problem']."""
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        _register_2026_tasks(available_tasks)     # restaurant_diner et al.
        _register_2026_scene_robot_class()        # skip the template's embedded robot
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
            # EXTRA_LOAD_ROOMS: comma-separated room TYPES to force-load on top of
            # the BDDL-derived set. Needed for novel-start nav slices whose starts
            # sit in rooms the activity never declares (hsf move_to_table_near_sofa:
            # corridor/entryway starts, but Christmas/bringing_water only load
            # kitchen+living+dining) — without this the robot spawns in void.
            _extra_rooms = [r.strip() for r in os.environ.get("EXTRA_LOAD_ROOMS", "").split(",")
                            if r.strip()]
            if _extra_rooms:
                relevant_rooms = sorted(set(relevant_rooms) | set(_extra_rooms))
                logger.info(f"[partial_scene_load] EXTRA_LOAD_ROOMS added {_extra_rooms} "
                            f"-> load_room_types={relevant_rooms}")
            cfg["scene"]["load_room_types"] = relevant_rooms

        # 3.9.x: generate_robot_config(robot_type, robot_name, task_name=, task_cfg=);
        # 3.7.x took only the keywords. Pick by signature so both trees work.
        import inspect as _inspect
        if "robot_type" in _inspect.signature(generate_robot_config).parameters:
            # 3.9.x keys gello's ROBOT_CONFIGS by LOWERCASE model id ("r1", "r1pro");
            # "R1Pro" KeyErrors. The dict it returns carries "type", which
            # env_base.py normalises to lowercase "model" on its own (it accepts
            # either, asserting only that both are not present at once).
            _robot_cfg = generate_robot_config("r1pro", "robot_r1",
                                               task_name=task_name, task_cfg=task_cfg)
        else:
            _robot_cfg = generate_robot_config(task_name=task_name, task_cfg=task_cfg)
        # OmniGibson >=3.9 moved robot-config construction into
        # Evaluator._build_robot_config, which this override replaces. __init__ then
        # reads self.robot_name / self.robot_eval_config to resolve the eval camera
        # roles, and _validate_robot_eval_config() hard-fails write_video runs when
        # they are empty. Populate them from upstream's own r1pro.yaml so the roles
        # cannot drift from the installed OmniGibson.
        if _OG_EVAL_NS == "omnigibson.eval":
            self.robot_name = _robot_cfg.get("name", "robot_r1")
            _eval_blk = _robot_cfg.pop("eval", None)
            if not _eval_blk:
                try:
                    from omnigibson.eval.evaluator import DEFAULT_ROBOT_CONFIG_PATH
                    _eval_blk = OmegaConf.to_container(
                        OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH), resolve=True
                    ).get("eval")
                except Exception as _e:  # noqa: BLE001
                    logger.warning(f"[robot] could not read r1pro.yaml eval block: {_e}")
                    _eval_blk = None
            if not _eval_blk:  # last resort: mirror ROBOT_CAMERA_NAMES
                _eval_blk = {"camera_sensor_names": {
                    role: name.split("::", 1)[1]
                    for role, name in ROBOT_CAMERA_NAMES["R1Pro"].items()
                }}
            self.robot_eval_config = _eval_blk

        cfg["robots"] = [_robot_cfg]
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        # HEAD_CAM_RES=720: bump the robot camera resolution (spawnval frame render
        # for VQA wants >224px; safe there — no policy consumes the obs). Applies to
        # all VisionSensors on the robot.
        _cam_res = os.environ.get("HEAD_CAM_RES")
        if _cam_res:
            sk = cfg["robots"][0].setdefault("sensor_config", {}).setdefault(
                "VisionSensor", {}).setdefault("sensor_kwargs", {})
            sk["image_height"] = sk["image_width"] = int(_cam_res)
            logger.info(f"[spawnval] HEAD_CAM_RES={_cam_res} -> robot cameras {_cam_res}x{_cam_res}")
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

        _disable_steam_fire_emitters()
        env = og.Environment(configs=cfg)

        # OmniGibson >=3.9: Evaluator.load_robot() resolves the robot by
        # `self.env.scene.object_registry("name", self.robot_name)` and returns
        # None -- not an error -- when the name does not match, so __init__ then
        # dies on `self.robot.name`. env_base only builds our configured robot
        # when the scene loaded none of its own (`if len(self.scene.robots) == 0`),
        # and the 2026 templates embed one (`robot_slwzxc`, class "Robot"), so the
        # name that ends up in the registry is not necessarily the one we asked
        # for. Take it from the robot the scene ACTUALLY has -- the same source
        # upstream's own `return self.env.robots[0]` fallback uses.
        if _OG_EVAL_NS == "omnigibson.eval":
            _scene_robots = list(getattr(env.scene, "robots", []) or [])
            if _scene_robots:
                _actual = _scene_robots[0].name
                _wanted = getattr(self, "robot_name", None)
                if _wanted is not None and _actual != _wanted:
                    # Our robot config was IGNORED -- env_base only builds it when
                    # the scene loaded none of its own. Adopting the scene's robot
                    # would run the policy against one with no eval cameras and
                    # none of our controllers, so fail loudly instead.
                    raise RuntimeError(
                        f"scene robot is {_actual!r} but we configured {_wanted!r}: the "
                        "template's embedded robot was loaded despite include_robots=False, "
                        "so our robot config never took effect. _register_2026_scene_robot_class() "
                        "should have prevented this."
                    )
                self.robot_name = _actual
            else:
                logger.warning("[robot] scene reports NO robots after env build")
            env._eval_robot_config = self.robot_eval_config

        from hydra.utils import instantiate
        env = instantiate(env_wrapper, env=env)
        return env

    def load_task_instance_from_tro(self, tro_path: str, open_doors_for_rooms=None, open_door_names=None) -> None:
        """Load a custom TRO state file instead of the default instance 0."""
        logger.info(f"Loading TRO: {tro_path}")
        with open(tro_path) as f:
            tro_state = recursively_convert_to_torch(json.load(f))

        for tro_key, tro_data in tro_state.items():
            if tro_key == "robot_poses":
                robot_model = _robot_model_key(self.robot, tro_data)
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
                robot_model = _robot_model_key(self.robot, tro_data)
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
        # Config-driven door names (from `doors_open`, passed by the caller) take
        # precedence; OPEN_DOOR_NAMES env stays as a manual override / legacy path.
        if open_door_names is not None:
            _door_names = [str(n).strip() for n in open_door_names if str(n).strip()]
        else:
            _door_names = [n.strip() for n in os.environ.get("OPEN_DOOR_NAMES", "").split(",") if n.strip()]
        _door_cats = [c.strip() for c in os.environ.get("OPEN_DOOR_CATEGORIES", "").split(",") if c.strip()]
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
                    # Only open doors that CONNECT the goal room to another INTERIOR
                    # room (interior path doors, in_rooms >= 2). Skip:
                    #  - single-room boundary/exterior doors (in_rooms < 2), e.g. the
                    #    garage→outside vehicle door garage_door_prdjyt_0.
                    #  - doors onto an EXTERIOR area (garden/lawn/...), e.g. the
                    #    kitchen→garden door — must stay at scene default (closed).
                    _ir = list(getattr(_d, "in_rooms", None) or [])
                    if len(_ir) >= 2 and not any(_is_exterior_room(_r) for _r in _ir):
                        _doors.add(_d)
                    elif len(_ir) >= 2:
                        logger.info(f"Skipping exterior-facing door {_d.name} (in_rooms={_ir})")
            for _cat in _door_cats:                      # explicit category override
                _doors |= (_reg("category", _cat) or set())
            for _nm in _door_names:                      # explicit name override
                _d = _reg("name", _nm)
                if _d is not None:
                    _doors.add(_d)
            for _door in _doors:
                try:
                    if _OpenState is not None and _OpenState in _door.states:
                        _door.states[_OpenState].set_value(True, fully=True)
                    else:
                        _door.set_joint_positions(_door.joint_upper_limits, drive=False)
                    logger.info(f"Opened door for nav precondition: {_door.name} (category={getattr(_door, 'category', '?')})")
                except Exception as _e:
                    logger.warning(f"Failed to open door {getattr(_door, 'name', '?')}: {_e}")
            # Capture the opened goal-door world xy positions. The nav success
            # criterion (run_single_instance) anchors on reaching the garage
            # DOOR, not on the garage room mask — the mask is the garage interior
            # (it starts ~0.85 m past the door) and its whole entrance edge is
            # wall, so distance-to-room mislabels both door-arrivals and wall-
            # grazes. The door object is the actual gate.
            # Extra opens happen AFTER the goal-door anchors are collected below,
            # so they can never be mistaken for a nav goal.
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

        # Traversability / context-consistency opens. Deliberately OUTSIDE the
        # `if _door_rooms or _door_names or _door_cats` block above and applied
        # after _goal_door_xys is populated, so these never become success anchors.
        _scene_model = str(getattr(self.env.scene, "scene_model", "") or "")
        _extra = list(_SCENE_ALWAYS_OPEN.get(_scene_model, []))
        _extra += [n.strip() for n in os.environ.get("EXTRA_OPEN_NAMES", "").split(",") if n.strip()]
        if _extra:
            try:
                from omnigibson.object_states import Open as _OpenState2
            except Exception:
                _OpenState2 = None
            for _nm in dict.fromkeys(_extra):
                _obj = self.env.scene.object_registry("name", _nm)
                if _obj is None:
                    logger.warning(f"[doors] extra-open object {_nm!r} not found in scene "
                                   f"{_scene_model!r} — skipping")
                    continue
                try:
                    # Mirror the context tour: prefer driving the joint to its LIMIT.
                    # Open.set_value only cracks a joint to its "open" threshold,
                    # which does not fully clear a pass-through.
                    _obj.set_joint_positions(_obj.joint_upper_limits, drive=False)
                    logger.info(f"[doors] extra-open (not a goal anchor): {_nm}")
                except Exception:
                    try:
                        if _OpenState2 is not None and _OpenState2 in _obj.states:
                            _obj.states[_OpenState2].set_value(True, fully=True)
                            logger.info(f"[doors] extra-open via Open state: {_nm}")
                        else:
                            logger.warning(f"[doors] could not open {_nm}: no joints and no Open state")
                    except Exception as _e2:
                        logger.warning(f"[doors] could not open {_nm}: {_e2}")
            if _doors:
                # Re-pin the restored F0 qpos so opening the doors (one physics
                # step below) doesn't nudge the robot off its spawn pose.
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


# ── Instruction forwarding to the websocket server ───────────────────────

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

    msg = {
        "reset": True,
        "instruction": subtask_config.get("instruction", ""),
        "task_category": subtask_config.get("task_category", ""),
        "task_type": subtask_config.get("task_type", ""),
        "target_instances": subtask_config.get("target_instances", []),
        "instance_name": instance_name,
    }
    ws._ws.send(ws._packer.pack(msg))
    logger.info(
        f"Sent reset+instruction to server: {subtask_config.get('instruction', '')!r}"
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


# ── Spawn validation (SPAWN_VALIDATE=1) ───────────────────────────────────────
# Reuses the proven pose validator from the paperback demo-augmentation replay
# (_validate_pose_bisector): place robot at the TRO spawn, re-pin the arm, let the
# base settle freely a few steps, then flag it if the robot is in contact with any
# scene object (furniture/wall — floors/ceilings excluded) OR the base drifts,
# sinks, or tips (an unstable edge-of-house spawn). This runs on the SAME env +
# TRO-load path as a real eval, so "valid here" == "valid at eval time". No policy
# rollout — the loop early-returns after this check.
def _spawn_tilt_angle(quat) -> float:
    """Angle (rad) between quat and its upright (pure-yaw) projection."""
    import omnigibson.utils.transform_utils as T
    if not th.is_tensor(quat):
        quat = th.tensor(quat, dtype=th.float32)
    z_angle = float(T.z_angle_from_quat(quat))
    upright = T.euler2quat(th.stack([quat.new_tensor(0.0), quat.new_tensor(0.0),
                                     quat.new_tensor(z_angle)]))
    tq = T.quat_distance(quat, upright)
    return float(2.0 * th.atan2(th.norm(tq[:3]), th.abs(tq[3])))


def _spawn_exclude_prims(scene) -> list:
    """Floor/ceiling objects to ignore in the contact test (everything else —
    furniture, walls, appliances — is a real obstacle for the base footprint)."""
    excl = []
    for obj in scene.objects:
        cat = (getattr(obj, "category", "") or "").lower()
        name = (getattr(obj, "name", "") or "").lower()
        if cat in {"floors", "floor", "ceilings", "ceiling"} or "floor" in name or "ceiling" in name:
            excl.append(obj)
    return excl


def _spawn_capture_arm(robot) -> list:
    """Capture trunk/arm/gripper qpos so we can re-pin them each settle step
    (an unsupported arm sagging under gravity would create spurious instability)."""
    groups = [robot.trunk_control_idx, robot.arm_control_idx["left"],
              robot.arm_control_idx["right"], robot.gripper_control_idx["left"],
              robot.gripper_control_idx["right"]]
    jp = robot.get_joint_positions()
    held = []
    for idx in groups:
        q = jp[idx]
        held.append((idx, q.clone() if hasattr(q, "clone") else th.tensor(q, dtype=th.float32)))
    return held


def _validate_spawn(evaluator, subtask_cfg: dict, instance_name: str,
                    out_dir: Path) -> bool:
    """Check the current (already-loaded) spawn for scene collision + stability.
    Writes <out_dir>/<instance_name>.json and returns spawn_ok."""
    # OmniGibson 3.9.x deleted the free collision helpers from utils.sim_utils
    # (check_collision / get_collisions / filter_collisions); the replacement is
    # RigidContactAPI.is_in_contact(scene_idx, query_set, with_set, ignore_set,
    # current_only) in utils.usd_utils, with different semantics (prim SETS and an
    # explicit scene index). Rather than guess at that mapping and emit
    # authoritative-looking but unverified spawn verdicts, treat the collision
    # component as UNAVAILABLE on 3.9.x: the stability checks (tilt / drift /
    # z-drop / velocities) still run and are reported normally.
    # This path is SPAWN_VALIDATE-only diagnostics -- the real rollout never
    # reaches it -- so 2026 evaluation is unaffected.
    try:
        from omnigibson.utils.sim_utils import check_collision
        _collision_supported = True
    except ImportError:
        _collision_supported = False

        def check_collision(*_a, **_kw):  # noqa: ANN002, ANN003
            return False
        logger.warning("[spawnval] check_collision unavailable on this OmniGibson "
                       "(>=3.9); reporting collision=unavailable, stability checks still run")

    robot = evaluator.robot
    scene = evaluator.env.scene
    settle = int(os.environ.get("SPAWN_SETTLE_STEPS", "30"))
    tols = {
        "tilt": float(os.environ.get("SPAWN_TILT_TOL", "0.10")),
        "drift": float(os.environ.get("SPAWN_DRIFT_TOL", "0.05")),
        "zdrop": float(os.environ.get("SPAWN_ZDROP_TOL", "0.03")),
        "linvel": float(os.environ.get("SPAWN_LINVEL_TOL", "0.05")),
        "angvel": float(os.environ.get("SPAWN_ANGVEL_TOL", "0.15")),
    }
    exclude = _spawn_exclude_prims(scene)
    held = _spawn_capture_arm(robot)

    # SPAWN_SAVE_FRAME=1: save the t=0 head-cam RGB BEFORE the settle probe
    # perturbs the state. Read the RAW VisionSensor (evaluator.obs is resized to
    # the policy's 224 by the obs pipeline regardless of sensor resolution — with
    # HEAD_CAM_RES=720 the sensor renders 720 but obs stays 224). A few warmup
    # renders first: the settle loop only steps physics, and the RTX denoiser
    # needs several renders to converge (same remedy as the video path).
    # PNG -> <log_path>/frames/<instance>.png. Makes spawnval double as the VQA
    # init-frame render.
    if os.environ.get("SPAWN_SAVE_FRAME", "0") == "1":
        try:
            import cv2
            sensor_name = ROBOT_CAMERA_NAMES["R1Pro"]["head"].split("::")[1]
            cam = robot.sensors[sensor_name]
            # The env wrapper (rgb_low_res_wrapper et al.) pins the sensors to 224
            # AFTER env creation, clobbering any sensor_config — so bump the sensor
            # here, render, save, restore. Runtime resolution changes are exactly
            # what the wrapper itself does, so this is supported.
            want = int(os.environ.get("HEAD_CAM_RES", "0"))
            old_hw = (int(cam.image_height), int(cam.image_width))
            if want and old_hw != (want, want):
                cam.image_height = cam.image_width = want
            for _ in range(max(1, int(os.environ.get("SPAWN_FRAME_WARMUP", "8")))):
                og.sim.render()
            so, _ = cam.get_obs()
            rgb = so["rgb"]
            rgb = (rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb))[..., :3]
            frames_dir = out_dir.parent / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frames_dir / f"{instance_name}.png"),
                        cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
            logger.info(f"[spawnval] saved frame {frames_dir / (instance_name + '.png')} "
                        f"({rgb.shape[1]}x{rgb.shape[0]})")
            if want and old_hw != (want, want):
                cam.image_height, cam.image_width = old_hw
        except Exception as e:
            logger.warning(f"[spawnval] frame save failed for {instance_name}: {e}")

    # Resolve carried-object prims up front. The robot is ALLOWED to be in contact
    # with what it carries, so these are EXCLUDED from the robot self-collision
    # check (otherwise a robot holding an object in front of it is spuriously
    # flagged); they're checked separately against the environment below.
    ns = subtask_cfg.get("novel_start") or {}
    carried_prims, carried_names = [], []
    for cname in (ns.get("carried_objects") or []):
        obj = evaluator.env.task.object_scope.get(cname)
        if obj is None:
            continue
        actual = obj.wrapped_obj if hasattr(obj, "wrapped_obj") else obj
        carried_prims.append(actual)
        carried_names.append(cname)

    pos0, _ = robot.get_position_orientation()
    cmd = th.tensor([float(pos0[0]), float(pos0[1]), float(pos0[2])], dtype=th.float32)
    # Free settle (re-pin arm each step; DON'T keep_still the base so a wedged /
    # unbalanced spawn is allowed to drift, sink, or tip).
    for _ in range(settle):
        for idx, q in held:
            robot.set_joint_positions(q, indices=idx)
        og.sim.step_physics()

    pos, ori = robot.get_position_orientation()
    drift = float(np.hypot(float(pos[0]) - float(cmd[0]), float(pos[1]) - float(cmd[1])))
    zdrop = float(float(cmd[2]) - float(pos[2]))
    tilt = _spawn_tilt_angle(ori)
    # Robot vs environment (floors/ceilings AND carried objects excluded).
    collided = _collision_supported and bool(check_collision(prims=robot, prims_exclude=exclude + carried_prims,
                                    step_physics=True))
    try:
        lin = float(th.norm(robot.get_linear_velocity()))
        ang = float(th.norm(robot.get_angular_velocity()))
    except Exception:
        lin = ang = 0.0

    # Carried object vs environment: a start where a held item clips a wall is a
    # broken spawn. Exclude floors/ceilings AND the robot (the item legitimately
    # touches the gripper). Recorded always (if SPAWN_CHECK_CARRIED=1); whether it
    # GATES spawn_ok is controlled by SPAWN_GATE_CARRIED (some "carried" items are
    # really table objects near the base — see find_carried).
    carried_collided = False
    if os.environ.get("SPAWN_CHECK_CARRIED", "1") == "1":
        for actual, cname in zip(carried_prims, carried_names):
            try:
                if bool(check_collision(prims=actual, prims_exclude=exclude + [robot],
                                        step_physics=False)):
                    carried_collided = True
            except Exception as e:
                logger.warning(f"[spawnval] carried check failed for {cname}: {e}")

    # robot_ok = robot itself is collision-free and stable (the nav-relevant test).
    robot_reason = ("collided" if collided else
                    "tilt" if tilt > tols["tilt"] else
                    "drift" if drift >= tols["drift"] else
                    "zdrop" if abs(zdrop) >= tols["zdrop"] else
                    "linvel" if lin >= tols["linvel"] else
                    "angvel" if ang >= tols["angvel"] else "ok")
    robot_ok = robot_reason == "ok"
    gate_carried = os.environ.get("SPAWN_GATE_CARRIED", "1") == "1"
    reason = (robot_reason if not robot_ok else
              ("carried_collided" if (carried_collided and gate_carried) else "ok"))
    spawn_ok = robot_ok and not (carried_collided and gate_carried)
    rec = {
        "instance": instance_name, "spawn_ok": spawn_ok, "reason": reason,
        "robot_ok": robot_ok, "robot_reason": robot_reason,
        "collided": collided, "carried_collided": carried_collided,
        # False here means "not checked" when collision_supported is False -- do
        # not read it as a passed collision test (OmniGibson >=3.9, see above).
        "collision_supported": _collision_supported,
        "tilt": round(tilt, 4), "drift": round(drift, 4), "zdrop": round(zdrop, 4),
        "linvel": round(lin, 4), "angvel": round(ang, 4),
        "settle_steps": settle, "tols": tols,
        "start_xy": ns.get("start_xy"), "start_yaw_deg": ns.get("start_yaw_deg"),
        "facing_vs_goal_deg": ns.get("facing_vs_goal_deg"),
        "source_room": ns.get("source_room"), "facing_bucket": ns.get("facing_bucket"),
        "carried_objects": carried_names,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{instance_name}.json", "w") as f:
        json.dump(rec, f, indent=2)
    logger.info(f"[spawnval] {instance_name}: {'OK ' if spawn_ok else 'BAD'} "
                f"reason={reason} collided={collided} carried={carried_collided} "
                f"tilt={tilt:.3f} drift={drift:.3f} zdrop={zdrop:.3f} "
                f"lin={lin:.3f} ang={ang:.3f}")
    return spawn_ok


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
        # Legacy fallback (was target_instances).
        if not ir_targets:
            ir_targets = list(subtask_cfg.get("target_instances", []) or [])
        ir_injector.update_target_instances(
            ir_targets, instance_name=instance_name,
        )

    # Load TRO to set initial state.
    evaluator.reset()
    # ── Door spec: DATA-DRIVEN via the config (not eval logic) ─────────────────
    # The generator writes the exact door object names to open into `doors_open`.
    # When present it is AUTHORITATIVE — the eval opens EXACTLY those (fully generic,
    # zero scene/task knowledge here) and skips the legacy room-based auto-logic +
    # OPEN_DOOR_*/NAV_DISABLE_* env overrides. Configs WITHOUT the field fall back
    # to the legacy behavior for backward compatibility.
    # Computed up-front (also used later for objnav setup), regardless of door path.
    _door_rooms = [c.get("room_instance") for c in goal_conditions
                   if c.get("type") == "in_room" and c.get("room_instance")]
    _obj_goals = [c for c in goal_conditions if c.get("type") == "near_object"]
    _cfg_doors = subtask_cfg.get("doors_open")
    if _cfg_doors is not None:
        logger.info(f"[doors] config-driven doors_open={list(_cfg_doors)} (authoritative)")
        evaluator.load_task_instance_from_tro(
            tro_file, open_doors_for_rooms=[], open_door_names=list(_cfg_doors))
    else:
        if _obj_goals and not _door_rooms and os.environ.get("NAV_DISABLE_AUTO_DOORS", "0") == "1":
            logger.info("[doors] (legacy) NAV_DISABLE_AUTO_DOORS=1 → skipping auto door-opening; OPEN_DOOR_NAMES only")
        elif _obj_goals and not _door_rooms:
            # OBJECT-target nav (fridge/table): open the START + TARGET room doors.
            _sm0 = getattr(getattr(evaluator.env, "scene", None), "_seg_map", None)
            if _sm0 is not None and hasattr(_sm0, "get_room_instance_by_point"):
                _rooms = set()
                try:
                    _sid = _obj_goals[0].get("target_object_id")
                    _to = evaluator.env.scene.object_registry("name", _sid)
                    if _to is not None:
                        _tp = _to.get_position_orientation()[0]
                        _tp = _tp.cpu().numpy() if hasattr(_tp, "cpu") else np.asarray(_tp)
                        _tr = _sm0.get_room_instance_by_point(_tp[:2])
                        if _tr:
                            _rooms.add(str(_tr))
                except Exception as _e:
                    logger.warning(f"[doors] target-room lookup failed: {_e}")
                try:
                    import json as _json
                    _rp = _json.load(open(tro_file))["robot_poses"]["R1Pro"][0]["position"]
                    _sr = _sm0.get_room_instance_by_point(np.asarray(_rp[:2], dtype=float))
                    if _sr:
                        _rooms.add(str(_sr))
                except Exception as _e:
                    logger.warning(f"[doors] start-room lookup failed: {_e}")
                _door_rooms = list(_rooms)
                logger.info(f"[doors] (legacy) object-target nav: opening doors for rooms {_door_rooms}")
        evaluator.load_task_instance_from_tro(tro_file, open_doors_for_rooms=_door_rooms)

    initial_positions = _capture_initial_positions(evaluator, goal_conditions)

    # Run the rollout.
    evaluator.reset()

    # Spawn-validation mode: check the first-frame spawn for scene collision /
    # edge-of-house instability, write a sidecar, and SKIP the policy rollout.
    if os.environ.get("SPAWN_VALIDATE") == "1":
        spawn_dir = metrics_path.parent / "spawnval"
        return _validate_spawn(evaluator, subtask_cfg, instance_name, spawn_dir)

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
        # head-only (448x448) unless VIDEO_INCLUDE_WRISTS=1 (then full 448x672 composite)
        _vid_res = (448, 672) if os.environ.get("VIDEO_INCLUDE_WRISTS", "0") == "1" else (448, 448)
        evaluator.video_writer = create_video_writer(
            fpath=video_name, resolution=_vid_res,
        )
    _metric_start(evaluator.metrics, evaluator.env)

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

    # Object-target nav success (fridge / kitchen-table): base within tol of the
    # target object, any step. Locate the target by SIM ID in the scene registry —
    # it may NOT be a BDDL task object (e.g. breakfast_table in carrying_in_groceries).
    objnav = None
    if _obj_goals:
        c0 = _obj_goals[0]
        sim_id = c0.get("target_object_id")
        txy = None
        _taabb = None
        try:
            _tobj = evaluator.env.scene.object_registry("name", sim_id)
            if _tobj is not None:
                _tp = _tobj.get_position_orientation()[0]
                _tp = _tp.cpu().numpy() if hasattr(_tp, "cpu") else np.asarray(_tp)
                txy = (float(_tp[0]), float(_tp[1]))
                from goal_checker import obj_xy_aabb as _aabb
                # In footprint mode the box defines the success region, so it must
                # not include articulated parts we open for traversability (the
                # bar's flip-up flap). Default to the structural base_link; a goal
                # condition may override with `footprint_links`.
                _fl = (c0.get("footprint_links")
                       if str(c0.get("distance_mode")) == "footprint" else None)
                if str(c0.get("distance_mode")) == "footprint" and _fl is None:
                    _fl = ["base_link"]
                # An explicit `footprint_box` in the goal condition wins over the
                # object's own AABB. The bar counter is 2.31 x 5.80 m, so even the
                # base_link box + 0.8 m tolerance accepts a 3.91 m-wide band of x
                # that reaches toward the kitchen -- standing on the SERVICE side,
                # near the kitchen, scored as "at the bar". The box lets the goal
                # be the dining-facing portion of the counter instead.
                _fbox = c0.get("footprint_box")
                if _fbox:
                    _taabb = ((float(_fbox[0][0]), float(_fbox[0][1])),
                              (float(_fbox[1][0]), float(_fbox[1][1])))
                    logger.info(f"[objnav] {sim_id}: EXPLICIT footprint_box "
                                f"x[{_taabb[0][0]:.2f},{_taabb[1][0]:.2f}] "
                                f"y[{_taabb[0][1]:.2f},{_taabb[1][1]:.2f}]")
                else:
                    _taabb = _aabb(_tobj, links=_fl)
                if _fl and _taabb is None and not _fbox:   # named links absent -> whole entity
                    logger.warning(f"[objnav] {sim_id}: links {_fl} not found; "
                                   "falling back to the WHOLE-entity AABB")
                    _taabb = _aabb(_tobj)
                elif _fl and not _fbox:
                    logger.info(f"[objnav] {sim_id}: footprint from links {_fl} "
                                f"-> x[{_taabb[0][0]:.2f},{_taabb[1][0]:.2f}] "
                                f"y[{_taabb[0][1]:.2f},{_taabb[1][1]:.2f}]")
        except Exception as _e:
            logger.warning(f"[objnav] could not locate target {sim_id}: {_e}")
        # Extra accepted targets (held_out relaxations, e.g. the other corner
        # booth). Which one was reached is recorded as `reached_candidate`.
        _alts = []
        for _an in (c0.get("alt_accept_object_ids") or []):
            try:
                _ao = evaluator.env.scene.object_registry("name", _an)
                if _ao is None:
                    logger.warning(f"[objnav] alt target {_an} not in scene; skipping")
                    continue
                _ap = _ao.get_position_orientation()[0]
                _ap = _ap.cpu().numpy() if hasattr(_ap, "cpu") else np.asarray(_ap)
                from goal_checker import obj_xy_aabb as _aabb2
                _afl = (c0.get("footprint_links") or ["base_link"]) \
                    if str(c0.get("distance_mode")) == "footprint" else None
                _ab = _aabb2(_ao, links=_afl) or _aabb2(_ao)
                _alts.append((_an, (float(_ap[0]), float(_ap[1])), _ab))
            except Exception as _e:
                logger.warning(f"[objnav] alt target {_an} lookup failed: {_e}")
        if txy is not None:
            objnav = ObjectNavSuccessTracker(
                txy, tol_m=float(c0.get("tol_m", 1.5)), target_name=sim_id,
                traj_every=int(os.environ.get("NAV_TRAJ_LOG_EVERY", "5")),
                # same-step room qualifier (hsf bookcase: target 0.74 m through a
                # wall from the corridor → distance alone can false-positive)
                require_room=c0.get("require_room"), seg_map=_seg_map,
                target_aabb=_taabb,
                distance_mode=str(c0.get("distance_mode", "centroid")),
                alt_targets=_alts,
            )
            logger.info(f"[objnav] target {sim_id} @ ({txy[0]:.2f},{txy[1]:.2f}) tol={objnav.tol}m"
                        f" mode={objnav.distance_mode}"
                        + (f" alt={[a[0] for a in _alts]}" if _alts else "")
                        + (f" require_room={objnav.require_room}" if objnav.require_room else ""))
            if str(c0.get("distance_mode")) == "footprint" and _taabb is None:
                logger.warning(f"[objnav] {sim_id}: footprint mode requested but AABB "
                               f"unavailable — FELL BACK to centroid distance")
        else:
            logger.error(f"[objnav] target {sim_id!r} not found in scene registry — cannot score success")

    terminated = truncated = False
    done = False
    while not done:
        terminated, truncated = evaluator.step()
        if terminated or truncated:
            done = True
        if nav.active or objnav is not None:
            try:
                _bxy = evaluator.robot.get_position_orientation()[0][:2]
                if nav.active:
                    nav.update(_bxy, evaluator.env._current_step)
                if objnav is not None:
                    objnav.update(_bxy, evaluator.env._current_step)
            except Exception:
                pass
        if config.write_video:
            for _ in range(_record_render_iters):
                og.sim.render()
            evaluator._write_video()
        if evaluator.env._current_step % 500 == 0:
            logger.info(f"  Step: {evaluator.env._current_step}")

    _metric_end(evaluator.metrics, evaluator.env)
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
    if (nav.active or objnav is not None) and _seg_map is not None:
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
            # Apply the spawn-satisfied guard HERE too, not just in the nav summary
            # dict: this value is what SubtaskGoalChecker turns into
            # `subtask_success`, so guarding only the summary left the headline
            # flag still reporting True for episodes that never moved.
            reached_object=((objnav.reached and not objnav.spawn_satisfied)
                            if objnav is not None else None),
        )
        subtask_success, subtask_results = checker.check_all()

    metrics: dict = {}
    metrics.update(_metric_results(evaluator.metrics, evaluator.env))
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
    elif objnav is not None:
        metrics["nav"] = objnav.to_metrics(_final_room, _final_base_xyz)
    # Surface the touched_target diagnostic (which candidate the agent was in
    # contact with at end-of-rollout, plus correct/wrong/none label) so reach
    # eval analysis can split outcomes per (slice, variant).
    if goal_conditions:
        touched_diag = getattr(checker, "touched_diag", None)
        if touched_diag is not None:
            metrics["touched_target"] = touched_diag

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

    with open(metrics_path / f"{instance_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Subtask success: {subtask_success}")

    if config.write_video:
        evaluator.video_writer = None
        logger.info(f"Saved video: {video_name}")

    return subtask_success


# ── Main (Hydra entry point) ─────────────────────────────────────────────

def main() -> None:
    # Hydra config tree shipped next to this script: OmniGibson 3.7.x's tree with
    # _target_s routed through og_compat, so it drives either install (3.9.x has
    # no hydra tree of its own).
    configs_dir = str(_THIS_DIR / "configs")
    if not os.path.isdir(configs_dir):
        logger.error(f"hydra config dir not found: {configs_dir}")
        sys.exit(1)
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
