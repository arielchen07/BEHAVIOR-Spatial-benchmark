"""Minimal policy server for the sim eval: a template for plugging in your method.

The policy here only holds the robot still (base velocity 0, every joint at the
position it had at the start of the episode). It shows the whole wire protocol
in one file, with no dependency beyond numpy, msgpack and websockets. Replace
`HoldPolicy` with your model; keep the server loop as is.

    python examples/hold_policy_server.py --port 8000
    # then, in the OmniGibson env:
    python bench/sim_eval/run_eval.py --eval-root $EVAL_ROOT --task reach_plate \\
        --slice held_out --host localhost --port 8000 --out runs/hold

Protocol (docs/protocol.md has the full observation/action contract):
  1. On connect the server sends a msgpack'd metadata dict.
  2. The client then sends msgpack'd dicts, one per message:
       * {"reset": True, "instruction": str, ...}  a new episode; NO reply is sent;
       * an observation dict                        reply {"action": float32[23]}.
  3. GET /healthz on the same port must return 200; the client polls it before
     connecting.
Numpy arrays travel as msgpack maps {"__ndarray__": True, "data", "dtype", "shape"}.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging

import msgpack
import numpy as np
import websockets.asyncio.server

logger = logging.getLogger("hold_policy_server")

HEAD_RGB_KEY = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
LEFT_WRIST_RGB_KEY = "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"
RIGHT_WRIST_RGB_KEY = "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
PROPRIO_KEY = "robot_r1::proprio"

# Where each part of the 23-dim action/state lives in the R1Pro proprio vector.
# Action layout: base_vel(3) trunk(4) arm_left(7) gripper_left(1) arm_right(7) gripper_right(1).
# OmniGibson 3.7.2 (2025 tasks) sends a 256-dim proprio, 3.9.2 (2026 diner tasks) a 61-dim one.
PROPRIO_SLICES = {
    256: dict(base_qvel=slice(253, 256), trunk=slice(236, 240), arm_left=slice(158, 165),
              gripper_left=slice(193, 195), arm_right=slice(197, 204), gripper_right=slice(232, 234)),
    61: dict(base_qvel=slice(0, 3), trunk=slice(53, 57), arm_left=slice(3, 10),
             gripper_left=slice(24, 26), arm_right=slice(28, 35), gripper_right=slice(49, 51)),
}


def robot_state(proprio: np.ndarray) -> np.ndarray:
    """23-dim state in action layout; each gripper is the sum of its two finger joints."""
    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    s = PROPRIO_SLICES[256 if proprio.shape[0] >= 256 else proprio.shape[0]]
    return np.concatenate([
        proprio[s["base_qvel"]], proprio[s["trunk"]], proprio[s["arm_left"]],
        [proprio[s["gripper_left"]].sum()], proprio[s["arm_right"]],
        [proprio[s["gripper_right"]].sum()],
    ]).astype(np.float32)


class HoldPolicy:
    """Replace with your method. `reset` starts an episode, `act` returns one action."""

    def __init__(self) -> None:
        self._hold = None

    def reset(self, instruction: str, **episode_info) -> None:
        logger.info("new episode: %r", instruction)
        self._hold = None

    def act(self, obs: dict) -> np.ndarray:
        # Images are uint8 HxWx3 (or x4; drop alpha). Extra privileged keys appear
        # under "policy_extras::..." only when the client runs with --with-ir.
        if self._hold is None:
            # Latch the first pose of the episode. Echoing the measured state every
            # step instead lets the targets follow gravity sag, and the arms creep.
            self._hold = robot_state(obs[PROPRIO_KEY])
            self._hold[0:3] = 0.0  # base velocity command
        return self._hold.copy()


# ── Wire format (matches OmniGibson's client) ─────────────────────────────────

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _handler(websocket, policy: HoldPolicy) -> None:
    packer = msgpack.Packer(default=_pack_array)
    await websocket.send(packer.pack({"server": "hold_policy_server"}))
    async for raw in websocket:
        msg = msgpack.unpackb(raw, object_hook=_unpack_array, strict_map_key=False)
        if msg.get("reset"):
            info = {k: v for k, v in msg.items() if k not in ("reset", "instruction")}
            policy.reset(msg.get("instruction", ""), **info)
            continue
        action = np.asarray(policy.act(msg), dtype=np.float32).reshape(23)
        await websocket.send(packer.pack({"action": action}))


async def serve(host: str, port: int) -> None:
    policy = HoldPolicy()
    async with websockets.asyncio.server.serve(
        lambda ws: _handler(ws, policy), host, port,
        compression=None, max_size=None, process_request=_health_check,
        # Match OmniGibson's client keepalive: scene resets can stall the sim's
        # side of the socket for longer than the 20 s websockets default.
        ping_interval=60, ping_timeout=300,
    ) as server:
        logger.info("serving on %s:%d", host, port)
        await server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args.host, args.port))
