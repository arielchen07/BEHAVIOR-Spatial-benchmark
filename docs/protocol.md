# Policy server protocol

The sim eval is split in two processes. The **client** (`bench/sim_eval/`) runs
OmniGibson, steps the scene and scores episodes. The **policy server** is your
method. They talk over one websocket, so the server can use any framework, Python
version or machine. `examples/hold_policy_server.py` implements everything below
in about 100 lines.

## Messages

All messages are msgpack. Numpy arrays are encoded as maps
`{"__ndarray__": True, "data": <bytes>, "dtype": <str>, "shape": <list>}`
(scalars: `{"__npgeneric__": True, "data", "dtype"}`).

| direction | when | payload | reply |
|---|---|---|---|
| server → client | right after connect | metadata dict (any content) | — |
| client → server | start of each episode | `{"reset": True, "instruction": str, "task_category": str, "task_type": str, "target_instances": list, "instance_name": str}` | **none** |
| client → server | every control step | observation dict | `{"action": float32[23]}` |

The server must also answer `GET /healthz` with HTTP 200 on the same port; the
client polls it before opening the websocket. `instruction` is the referring
expression the policy must follow. The other reset fields are for logging; a
method that uses `target_instances` is using privileged information.

## Observation

| key | value |
|---|---|
| `robot_r1::robot_r1:zed_link:Camera:0::rgb` | head camera, 224×224 uint8, RGB or RGBA |
| `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb` | left wrist camera, 224×224 |
| `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb` | right wrist camera, 224×224 |
| `robot_r1::proprio` | R1Pro proprioception: 256-dim on OmniGibson 3.7.2, 61-dim on 3.9.2 |

Other keys emitted by OmniGibson may be present; ignore what you don't use. The
proprio slices for each part of the robot are in `examples/hold_policy_server.py`
(`PROPRIO_SLICES`).

Navigation episodes use the same messages. The paper's navigation policies also take
the scene's context tour (a fixed, ordered set of frames per scene); the client does
not send it, so a server that uses it loads it from
`$DATA/action/train/navigation_context/<scene>/` at startup.

### Privileged IR keys (`run_eval.py --with-ir` only)

With `--with-ir` the client projects the instructed target's 3D box into the
camera from live sim state and adds:

| key | value |
|---|---|
| `policy_extras::object_center_xy` | int32 `[[cx, cy], ...]` in 224-px head-image coordinates; `[[-1, -1]]` when the target is off-screen |
| `policy_extras::object_center_left_wrist_xy` / `..._right_wrist_xy` | same for the wrist cameras (`--with-wrist-ir`) |
| `policy_extras::object_center_first_image` | frame-0 anchor (`--with-first-frame`) |

This is the oracle-IR setting from the paper. Results that use these keys must
be reported as privileged. The renderers that turn them into the dot / heatmap
images our IR policies were trained on are in `bench/sim_eval/privileged_ir/`.

## Action

`float32[23]`, in this order:

| index | meaning |
|---|---|
| 0:3 | base velocity command (x, y, yaw) |
| 3:7 | trunk joint positions |
| 7:14 | left arm joint positions |
| 14 | left gripper (sum of the two finger joint positions) |
| 15:22 | right arm joint positions |
| 22 | right gripper |

## Timing

The client sends an observation every control step and waits for the reply. A
server that runs inference inside that request (like the example) therefore
pauses the simulation while it thinks: latency costs wall-clock time, not task
performance.

The paper's VLA results used a different server design (Appendix F.3): it answers every step
immediately from a buffered action chunk and runs inference in the background,
replacing the buffer when a new chunk is ready. The sim then never waits for the
model, so inference latency does affect behavior: actions are executed against
observations that are a few hundred ms old. If that server stalls, the robot
keeps replaying stale actions and the episode usually ends as `none`. Both
designs are valid; state which one you used when reporting results.
