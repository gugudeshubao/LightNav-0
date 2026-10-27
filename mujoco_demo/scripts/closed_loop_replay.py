"""Isolated LightNav -> Soma pure DWA -> MuJoCo closed loop, never hardware.

Physics pauses during inference: this is an ideal synchronous simulation, not
real-time latency/freshness acceptance or the production instruction-task stack.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import select
import subprocess
import sys
import time
from pathlib import Path


def planner_step(process, request):
    process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
    process.stdin.flush()
    if not select.select([process.stdout], [], [], 10)[0]:
        raise RuntimeError("pure DWA process response timeout")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("pure DWA process exited")
    reply = json.loads(line)
    if not reply.get("ok") or reply.get("hardware_actuation") is not False:
        raise RuntimeError(f"DWA refused simulation input: {reply}")
    command = reply["command"]
    if len(command) != 3 or not all(math.isfinite(v) for v in command):
        raise RuntimeError("invalid DWA command")
    if abs(command[0]) > 0.6 + 1e-9 or abs(command[1]) > 1e-9 or abs(command[2]) > math.pi / 4 + 1e-9:
        raise RuntimeError("DWA command exceeds fixed simulation bounds")
    return reply


def stop_child(process, *, eof=False):
    if process is None:
        return None, False
    if process.poll() is None:
        if eof:
            process.stdin.close()
        else:
            process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return process.returncode, True
    return process.returncode, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "hf"), default="native")
    parser.add_argument("--flash-source", type=Path)
    parser.add_argument("--daemon", type=Path)
    parser.add_argument("--hf-source", type=Path)
    parser.add_argument("--hf-model", type=Path)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--points", type=int, choices=(10, 5), default=10)
    parser.add_argument("--instruction", default="go to the television")
    args = parser.parse_args()
    if not 1 <= args.steps <= 256 or not args.instruction.strip():
        parser.error("steps must be in [1,256] and instruction nonempty")
    if not args.output_dir.is_absolute() or args.output_dir.exists():
        parser.error("output-dir must be a fresh absolute directory")
    paths = [args.planner]
    if args.backend == "native":
        if not args.daemon or not args.flash_source:
            parser.error("native backend requires daemon and flash-source")
        paths += [args.daemon, args.flash_source / "scripts/lightnav_ws_bridge.py"]
    else:
        if not args.hf_source or not args.hf_model:
            parser.error("hf backend requires hf-source and hf-model")
        paths += [args.hf_source / "lightnav/tracking.py", args.hf_model / "config.json"]
    for path in paths:
        if not path.is_absolute() or not path.is_file():
            parser.error("explicit existing absolute binaries, source and assets are required")
    environment = os.environ.copy()
    required = ("LIGHTNAV_MODEL_DIR", "LIGHTNAV_EDGELLM_ENGINE_DIR",
                "LIGHTNAV_EDGELLM_VISUAL_ENGINE_DIR", "LIGHTNAV_EDGELLM_PLUGIN_PATH")
    for name in required if args.backend == "native" else ():
        if not environment.get(name):
            parser.error(f"missing {name}")
    for name in list(environment):
        if name.startswith(("AGENT_", "FEI_AGENT_")):
            environment.pop(name)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco
    import numpy as np
    from PIL import Image
    from vln_mujoco.model import PHYSICS_DT, scene_path
    from vln_mujoco.robots.turtlebot import TurtleBotBackend

    if args.backend == "native":
        sys.path.insert(0, str(args.flash_source))
        from scripts.lightnav_ws_bridge import FlashIpcClient

    args.output_dir.mkdir(parents=True, exist_ok=False)
    socket_path = args.output_dir / "infer.sock"
    if len(str(socket_path).encode()) >= 104:
        raise ValueError("output path is too long for a Unix socket")
    environment["AGENT_LLM_ENGINE_DIR"] = str(args.output_dir / "absent-agent")
    environment["LIGHTNAV_REQUIRE_READY"] = "1"
    report = {
        "schema": "lightnav-dwa-mujoco-closed-loop/v1", "hardware_actuation": False,
        "physics_paused_during_inference": True, "production_task_stack": False,
        "points": args.points, "instruction": args.instruction, "requested_steps": args.steps,
        "backend": args.backend,
        "obstacles": "360 horizontal simulator rays at base_z+0.18m, <=4m; sampled surfaces, not a full safety map",
        "planning_footprint": "unchanged DWA Go2 defaults, not calibrated to TurtleBot geometry",
        "collision_guard": "TurtleBot kinematic simulator guard remains enabled as independent final stop",
        "versions": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "pillow")},
        "scene_sha256": hashlib.sha256(scene_path().read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "planner_sha256": hashlib.sha256(args.planner.read_bytes()).hexdigest(),
        "daemon_sha256": hashlib.sha256(args.daemon.read_bytes()).hexdigest() if args.backend == "native" else None,
        "engine_paths": {name: environment[name] for name in required} if args.backend == "native" else {},
        "observations": [], "controls": [], "termination": "step_budget",
    }
    daemon = planner = client = renderer = None
    start = time.monotonic()
    with (args.output_dir / "daemon.log").open("w") as daemon_log, (
        args.output_dir / "planner.log"
    ).open("w") as planner_log:
        try:
            agent = None
            if args.backend == "native":
                daemon = subprocess.Popen([str(args.daemon), "--socket", str(socket_path)],
                                          env=environment, stdout=daemon_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 120
                while not socket_path.exists():
                    if daemon.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("isolated model host failed readiness")
                    time.sleep(0.1)
                client = FlashIpcClient(socket_path, args.output_dir.name, 30)
                client.connect()
                client.reset()
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["VLN_EVAL_TRAJ_TOP1"] = "1"
                os.environ["VLN_EVAL_TEMPERATURE"] = "0"
                sys.path.insert(0, str(args.hf_source))
                import torch
                import transformers
                from lightnav.tracking import build_tracking_agent

                if not torch.cuda.is_available() or transformers.__version__ != "5.8.0":
                    raise RuntimeError("HF reference requires CUDA and transformers 5.8.0; no fallback")
                agent = build_tracking_agent(model_path=str(args.hf_model), backend="hf",
                                             device="cuda", task_key="vlnce", max_new_tokens=5)
                agent.engine.bundle.model.to(dtype=torch.bfloat16)
                agent.reset(instruction=args.instruction)
                report["hf_reference"] = {"dtype": "bfloat16", "torch": torch.__version__,
                                           "transformers": transformers.__version__,
                                           "model": str(args.hf_model), "source": str(args.hf_source),
                                           "preprocessing": "unchanged upstream", "vocabulary": "upstream traj-top1"}
            planner = subprocess.Popen([str(args.planner)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=planner_log,
                                       text=True, bufsize=1)
            robot = TurtleBotBackend()
            renderer = mujoco.Renderer(robot.model, height=270, width=480)
            base_body = robot.model.body("base_link").id
            angles = np.arange(360) * (math.pi / 180)
            directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(360)))
            geom_id = np.empty(1, dtype=np.int32)
            televisions = [i for i in range(robot.model.nbody)
                           if (mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("television_")]
            assert abs(10 * PHYSICS_DT - 0.05) < 1e-9
            for index in range(args.steps):
                state = robot.state()
                x, y, _z, yaw = state.pose
                capture = (x, y, yaw)
                renderer.update_scene(robot.data, camera=robot.camera_name)
                rgb = renderer.render()
                filename = f"frame-{index:04d}.png"
                Image.fromarray(rgb).save(args.output_dir / filename)
                infer_started = time.monotonic()
                if client is not None:
                    prediction = client.step(request_id=index + 1, instruction=args.instruction,
                                             task_key="vlnce", capture_stamp_ns=1 + index * 250_000_000,
                                             width=480, height=270, rgb=rgb.tobytes())
                else:
                    agent.observe(rgb.copy())
                    points, raw, latency = agent.predict_waypoints(args.instruction, task_type="vlnce_traj")
                    points = np.asarray(points)
                    if points.shape != (10, 3) or not np.isfinite(points).all():
                        raise RuntimeError("invalid HF reference trajectory")
                    prediction = {
                        "ok": True, "request_id": index + 1, "raw_text": raw,
                        "raw_output_token_ids": agent.engine.bundle.processor.tokenizer.encode(raw, add_special_tokens=False),
                        "stop": bool(np.all(np.abs(points) <= 5e-3)), "latency_ms": latency,
                        "local_trajectory": [{"x": float(p[0]), "y": float(p[1]), "yaw": float(p[2])} for p in points],
                    }
                report["observations"].append({
                    "file": filename, "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "simulation_time_s": float(robot.data.time), "pose_xyz_yaw": state.pose,
                    "prediction": prediction, "inference_wall_ms": (time.monotonic() - infer_started) * 1000,
                    "nearest_tv_body_xy_distance_proxy_m": min(
                        (math.hypot(x - robot.data.xpos[i, 0], y - robot.data.xpos[i, 1]) for i in televisions), default=None),
                })
                for tick in range(5):
                    current = robot.state()
                    cx, cy, cz, cyaw = current.pose
                    velocity, angular = current.velocity
                    origin = np.array([cx, cy, cz + 0.18], dtype=np.float64)
                    obstacles = []
                    for direction in directions:
                        distance = mujoco.mj_ray(robot.model, robot.data, origin, direction,
                                                 None, True, base_body, geom_id)
                        if 0 <= distance <= 4:
                            obstacles.append([float(cx + distance * direction[0]),
                                              float(cy + distance * direction[1])])
                    command = planner_step(planner, {
                        "simulation_only": True, "capture_pose": capture,
                        "state": [cx, cy, cyaw, velocity, 0, angular],
                        "local_trajectory": prediction["local_trajectory"],
                        "stop": prediction["stop"], "point_count": args.points,
                        "obstacles": obstacles,
                    })
                    for _ in range(10):
                        robot.step((command["command"][0], command["command"][2]))
                    after = robot.state()
                    report["controls"].append({"decision": index + 1, "tick": tick,
                                               "pose_before": current.pose, "pose_after": after.pose,
                                               "planner": command, "obstacles": obstacles,
                                               "guard": after.telemetry})
                    if prediction["stop"]:
                        break
                if prediction["stop"]:
                    report["termination"] = "model_stop_not_independently_verified_arrival"
                    break
            report["final_pose"] = robot.state().pose
            report["final_guard"] = robot.state().telemetry
            report["simulated_time_s"] = float(robot.data.time)
        except Exception as error:  # noqa: BLE001 - record failure and reap only our child processes
            report["error"] = str(error)
            report["termination"] = "execution_error"
        finally:
            if renderer is not None:
                renderer.close()
            if client is not None:
                try:
                    client.reset()
                except Exception as error:  # noqa: BLE001 - cleanup failure is recorded, never a pass
                    report["reset_error"] = str(error)
                client.close()
            report["planner_returncode"], report["planner_forced_kill"] = stop_child(planner, eof=True)
            report["daemon_returncode"], report["daemon_forced_kill"] = stop_child(daemon)
            report["socket_removed"] = not socket_path.exists()
    report["wall_time_s"] = time.monotonic() - start
    report["execution_passed"] = (
        bool(report["observations"]) and not report.get("error") and not report.get("reset_error")
        and report["planner_returncode"] == 0 and (args.backend == "hf" or report["daemon_returncode"] == 0)
        and report["socket_removed"] and not report["planner_forced_kill"] and not report["daemon_forced_kill"]
    )
    report["navigation_success_claimed"] = False
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("observations", "controls")}, indent=2))
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
