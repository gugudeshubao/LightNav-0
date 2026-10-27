"""Continuous MuJoCo physics + persistent Soma C++ cloud client + C++ DWA.

No robot transport. Rendering/control may cost wall time; inference never
deliberately pauses physics. Report real-time factor rather than assuming it.
"""
import argparse
import json
import math
import os
import select
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from closed_loop_replay import planner_step, stop_child
from cloud_fault_relay import FaultRelay


def exchange(process, request):
    exchange_started_ns = time.monotonic_ns()
    process.stdin.write(json.dumps({"simulation_only": True, **request}) + "\n")
    process.stdin.flush()
    if not select.select([process.stdout], [], [], 3)[0]:
        raise RuntimeError("C++ cloud client pipe timeout")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("C++ cloud client exited")
    reply = json.loads(line)
    if reply.get("hardware_actuation") is not False:
        raise RuntimeError("not a simulation response")
    reply["probe_exchange_started_ns"] = exchange_started_ns
    reply["probe_reply_received_ns"] = time.monotonic_ns()
    return reply


def timing_breakdown(job, reply, consumed_ns):
    """Same-host monotonic diagnostics only; never changes admission/deadlines."""
    captured = job["stamp"]
    submitted = job["submitted_ns"]
    started = reply["probe_exchange_started_ns"]
    received = reply["probe_reply_received_ns"]
    if not captured <= submitted <= started <= received <= consumed_ns:
        raise RuntimeError("non-monotonic simulation timing evidence")
    return {"capture_to_submit_ms": (submitted-captured)/1e6,
            "executor_wait_ms": (started-submitted)/1e6,
            "stdio_roundtrip_ms": (received-started)/1e6,
            "reply_to_consume_ms": (consumed_ns-received)/1e6,
            "capture_to_consume_ms": (consumed_ns-captured)/1e6}


def usable(active, epoch, now_ns):
    return bool(active and active["epoch"] == epoch and active["reply"].get("ok")
                and active["reply"]["produced_at_ns"] <= now_ns
                < active["reply"]["deadline_ns"])


def accept(job, reply, epoch, now_ns):
    if job["epoch"] != epoch or not reply.get("ok"):
        return None
    if reply.get("request_id") != job["request_id"] or reply.get("source_stamp_ns") != job["stamp"]:
        raise RuntimeError("client result identity mismatch")
    result = {**job, "reply": reply}
    return result if usable(result, epoch, now_ns) else None


def retry_reset(failures, budget):
    """Only explicit bounded recovery may reopen a failed session."""
    return 0 < failures <= budget


def check_capture_budget(reply, budget_ms):
    if reply.get("capture_age_limit_ms") != budget_ms:
        raise RuntimeError("simulation client capture budget mismatch")
    if (reply.get("ok") and "deadline_ns" in reply
            and reply["deadline_ns"] - reply["source_stamp_ns"] != budget_ms * 1_000_000):
        raise RuntimeError("simulation client moved the capture deadline")


def planning_diagnostics(planned, position, expired):
    """Describe existing decisions without changing planner inputs or commands."""
    if planned is None:
        return {"reason": "no_valid_path"}
    if expired:
        reason = "expired_after_planning"
    elif any(abs(v) > 1e-9 for v in planned["command"]):
        reason = "moving"
    elif planned["mode"] == "model_stop":
        reason = "model_stop"
    elif planned["local_path_arrived"]:
        reason = "heading_complete" if planned["mode"] in ("heading", "short_path_heading") else "local_reference_complete"
    elif planned["obstacle_blocked"]:
        reason = "planner_blocked"
    else:
        reason = "unclassified_zero"
    result = {"reason": reason, "planner": planned}
    path = planned["reference"]
    if path:
        end = path[-1]
        dx, dy = end[0] - position[0], end[1] - position[1]
        result["endpoint_distance_m"] = math.hypot(dx, dy)
        result["endpoint_forward_projection_m"] = dx * math.cos(end[2]) + dy * math.sin(end[2])
    return result


def run(args):
    import mujoco
    import numpy as np

    from vln_mujoco.robots.turtlebot import TurtleBotBackend

    report = {"schema": "lightnav-cloud-continuous-simulation/v1",
              "hardware_actuation": False, "physics_paused_during_inference": False,
              "navigation_success_claimed": False, "production_task_stack": False,
              "capture_age_limit_ms": args.capture_age_ms,
              "finish_short_path_heading": args.finish_short_path_heading,
              "observations": [], "controls": [], "events": [],
              "obstacles": "360 horizontal simulator rays <=4m, not full scene geometry",
              "footprint": "unchanged Go2 DWA defaults; not calibrated TurtleBot",
              "instruction": args.instruction, "switch_instruction": args.switch_instruction,
              "gap_injection_enabled": args.inject_gap,
              "reset_retry_budget": args.reset_retries,
              "transport_drop_seconds": args.transport_drop_seconds,
              "transport_fault_scope": "dedicated loopback TCP relay closes real connections; no device network changes",
              "gap_injection": "withhold captures for 1 second, not a real network disconnect"}
    client = planner = renderer = robot = relay = None
    pool = ThreadPoolExecutor(max_workers=1)
    future = None
    started = time.monotonic()
    try:
        gateway = args.gateway
        if args.transport_drop_seconds:
            relay = FaultRelay()
            gateway = f"http://127.0.0.1:{relay.port}"
        command = [str(args.client), gateway]
        if args.session_journal:
            command.append(str(args.session_journal))
        client = subprocess.Popen(command, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        planner = subprocess.Popen([str(args.planner)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True, bufsize=1)
        robot = TurtleBotBackend()
        renderer = mujoco.Renderer(robot.model, height=270, width=480)
        base_body = robot.model.body("base_link").id
        angles = np.arange(360) * math.pi / 180
        directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(360)))
        geom_id = np.empty(1, dtype=np.int32)
        epoch, request_id = 1, 0
        instruction = args.instruction
        active = None
        pending = []
        need_reset, ready, switched = True, False, False
        reset_failures, reset_after = 0, 0.0
        job = None
        started = time.monotonic()
        next_capture = started
        transport_down = False
        while time.monotonic() - started < args.seconds:
            tick_start = time.monotonic()
            elapsed = tick_start - started
            down = bool(relay and args.seconds / 3 <= elapsed
                        < args.seconds / 3 + args.transport_drop_seconds)
            if down != transport_down:
                relay.set_down(down)
                transport_down = down
                report["events"].append({"event": "transport_down" if down else "transport_restored",
                                         "wall_s": elapsed, "epoch": epoch})
            if args.switch_instruction and not switched and elapsed >= args.seconds / 2:
                epoch += 1
                instruction = args.switch_instruction
                switched, need_reset, ready = True, True, False
                active, pending = None, []
                report["events"].append({"event": "task_switch", "wall_s": elapsed, "epoch": epoch})
            if future is not None and future.done():
                reply = future.result()
                check_capture_budget(reply, args.capture_age_ms)
                if job["action"] == "reset":
                    # Both use CLOCK_MONOTONIC on the same Linux host.
                    if abs(time.monotonic_ns() - reply["local_now_ns"]) > 1_000_000_000:
                        raise RuntimeError("client and simulator monotonic clocks disagree")
                    if job["epoch"] == epoch:
                        if not reply["ok"]:
                            reset_failures += 1
                            report["events"].append({"event": "reset_failed", "wall_s": elapsed,
                                                     "epoch": epoch, "error": reply.get("error"),
                                                     "attempt": reset_failures})
                            active, pending = None, []
                            ready, need_reset = False, True
                            if not retry_reset(reset_failures, args.reset_retries):
                                raise RuntimeError(reply.get("error", "reset failed"))
                            reset_after = time.monotonic() + 0.5
                        else:
                            ready, need_reset = True, False
                            reset_failures = 0
                            report["events"].append({"event": "reset_ready", "wall_s": elapsed,
                                                     "epoch": epoch})
                else:
                    consumed_ns = time.monotonic_ns()
                    adopted = accept(job, reply, epoch, consumed_ns)
                    report["observations"].append({**job, "reply": reply, "adopted": adopted is not None,
                                                   "timing": timing_breakdown(job, reply, consumed_ns),
                                                   "received_wall_s": elapsed})
                    if adopted:
                        active = adopted
                    elif job["epoch"] == epoch:
                        # No history continuation after unknown/late result.
                        active, pending = None, []
                        epoch += 1
                        ready, need_reset = False, True
                future = None
            if future is None and need_reset and time.monotonic() >= reset_after:
                job = {"action": "reset", "epoch": epoch}
                future = pool.submit(exchange, client, {"action": "reset", "session_id": f"sim-{epoch}"})
            gap = args.inject_gap and args.seconds * 0.75 <= elapsed < args.seconds * 0.75 + 1
            if ready and not gap and tick_start >= next_capture:
                stamp = time.monotonic_ns()
                state = robot.state()
                renderer.update_scene(robot.data, camera=robot.camera_name)
                rgb = renderer.render()
                path = args.output_dir / f"frame-{stamp}.rgb"
                path.write_bytes(rgb.tobytes())
                pending.append({"rgb_path": str(path), "width": 480, "height": 270, "stamp_ns": stamp,
                                "pose": [state.pose[0], state.pose[1], state.pose[3]]})
                next_capture = tick_start + 0.25  # actual capture cadence; no invented catch-up frames
            if len(pending) > 8:
                raise RuntimeError("bounded observation history overflow")
            if future is None and ready and pending:
                request_id += 1
                latest = pending[-1]
                job = {"action": "infer", "epoch": epoch, "request_id": request_id,
                       "stamp": latest["stamp_ns"], "capture_pose": latest["pose"],
                       "capture_wall_s": (latest["stamp_ns"] / 1e9) - started,
                       "frames": [p["rgb_path"] for p in pending],
                       "submitted_ns": time.monotonic_ns()}
                future = pool.submit(exchange, client, {
                    "action": "infer", "request_id": request_id, "instruction": instruction,
                    "local_capture_stamp_ns": latest["stamp_ns"], "observations": pending})
                pending = []
            current = robot.state()
            x, y, z, yaw = current.pose
            velocity, angular = current.velocity
            command = [0.0, 0.0, 0.0]
            planned = None
            expired_after_planning = False
            valid = usable(active, epoch, time.monotonic_ns())
            if valid:
                origin = np.array([x, y, z + 0.18], dtype=np.float64)
                obstacles = []
                for direction in directions:
                    distance = mujoco.mj_ray(robot.model, robot.data, origin, direction,
                                             None, True, base_body, geom_id)
                    if 0 <= distance <= 4:
                        obstacles.append([float(x + distance * direction[0]),
                                          float(y + distance * direction[1])])
                planned = planner_step(planner, {
                    "simulation_only": True, "capture_pose": active["capture_pose"],
                    "state": [x, y, yaw, velocity, 0, angular],
                    "local_trajectory": active["reply"]["local_trajectory"],
                    "stop": active["reply"]["stop"], "point_count": 10, "obstacles": obstacles,
                    "finish_short_path_heading": args.finish_short_path_heading})
                if usable(active, epoch, time.monotonic_ns()):
                    command = planned["command"]
                else:
                    expired_after_planning = True
            # Recheck during substeps: a slow planner never extends validity.
            for _ in range(10):
                if not usable(active, epoch, time.monotonic_ns()):
                    command = [0.0, 0.0, 0.0]
                    expired_after_planning = planned is not None
                robot.step((command[0], command[2]))
            report["controls"].append({
                "wall_s": time.monotonic() - started, "simulation_s": float(robot.data.time),
                "epoch": epoch, "command": command, "valid_before_planning": valid,
                "capture_gap_injected": bool(gap), "pose": robot.state().pose,
                "guard": robot.state().telemetry,
                "decision": planning_diagnostics(planned, (x, y), expired_after_planning),
                "request_id": active["request_id"] if active else None})
            time.sleep(max(0, 0.05 - (time.monotonic() - tick_start)))
            # Atomic progress evidence enables external tunnel-fault timing;
            # it is not an actuator or an instruction input.
            progress = args.output_dir / "progress.tmp"
            progress.write_text(json.dumps({"wall_s": elapsed, "epoch": epoch, "ready": ready,
                                            "observations": len(report["observations"])}))
            progress.replace(args.output_dir / "progress.json")
        report["simulated_time_s"] = float(robot.data.time)
        report["final_pose"] = robot.state().pose
        report["loop_completed"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
    finally:
        report["wall_time_s"] = time.monotonic() - started
        if robot is not None:
            # Retain partial physical progress even when recovery exhausts its
            # budget; an error must not turn actual elapsed physics into zero.
            report["simulated_time_s"] = float(robot.data.time)
            report["final_pose"] = robot.state().pose
        pool.shutdown(wait=True)
        if future is not None:
            try:
                report["drained_unconsumed"] = {"job": job, "reply": future.result()}
            except Exception as exc:  # noqa: BLE001
                report["drain_error"] = str(exc)
        if client is not None and client.poll() is None:
            try:
                report["close"] = exchange(client, {"action": "close"})
            except Exception as exc:  # noqa: BLE001
                report["close_error"] = str(exc)
        if renderer is not None:
            renderer.close()
        report["client_exit"], report["client_killed"] = stop_child(client, eof=True)
        report["planner_exit"], report["planner_killed"] = stop_child(planner, eof=True)
        if relay is not None:
            relay.close()
    report["real_time_factor"] = report.get("simulated_time_s", 0) / report["wall_time_s"]
    report["execution_passed"] = bool(report.get("loop_completed") and report.get("close", {}).get("ok")
                                      and report["client_exit"] == report["planner_exit"] == 0
                                      and not report["client_killed"] and not report["planner_killed"])
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ("observations", "controls")}))
    return 0 if report["execution_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--gateway", default="http://127.0.0.1:19882")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--capture-age-ms", type=int, choices=(500, 650, 800), default=500,
                        help="Must match the separately compiled simulation client")
    parser.add_argument("--instruction", default="go to the television")
    parser.add_argument("--finish-short-path-heading", action="store_true",
                        help="Simulation only: finish body yaw on completed paths within 8 cm")
    parser.add_argument("--switch-instruction")
    parser.add_argument("--inject-gap", action="store_true")
    parser.add_argument("--reset-retries", type=int, default=0)
    parser.add_argument("--transport-drop-seconds", type=float, default=0)
    parser.add_argument("--session-journal", type=Path)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 60 or not args.instruction.strip():
        parser.error("seconds must be 5..60 and instruction nonempty")
    if not 0 <= args.reset_retries <= 20:
        parser.error("reset-retries must be 0..20; default is no retry")
    if not 0 <= args.transport_drop_seconds <= 5 or (args.transport_drop_seconds and args.seconds < 10):
        parser.error("transport-drop-seconds must be 0..5; fault run must be at least 10 s")
    if args.gateway != "http://127.0.0.1:19882":
        parser.error("probe requires the dedicated loopback SSH tunnel")
    if any(not p.is_absolute() or not p.is_file() for p in (args.client, args.planner)):
        parser.error("explicit existing absolute client and planner required")
    if not args.output_dir.is_absolute() or args.output_dir.exists():
        parser.error("output-dir must be new and absolute")
    args.output_dir.mkdir(parents=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
