"""Exercise the actual web-demo loop against an explicit loopback model server.

No web UI or hardware transport is started. Physics/rendering run independently
of inference; this is the demo MPC path, not production Soma DWA assembly.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse


async def run(args):
    from vln_mujoco.server import VlnMujocoServer
    from vln_mujoco.timing import continuity_summary

    app = VlnMujocoServer(default_vln_server=args.vln_server)
    samples = []
    report = {"hardware_actuation": False, "physics_paused_during_inference": False,
              "controller": "demo MPC, not Soma DWA", "navigation_success_claimed": False,
              "instruction": args.instruction}
    try:
        await app.on_startup(None)
        app.vln.start(args.instruction)
        app.auto_owner = object()
        app.auto_mode = "vln"
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            if app.update_task.done():
                await app.update_task
                raise RuntimeError("demo loop exited unexpectedly")
            snapshot = app.snapshot(None)
            samples.append({"wall_elapsed_s": time.monotonic() - start,
                            "path_deadline_remaining_s": (
                                app._vln_path_deadline - time.monotonic()
                                if app._vln_path_deadline is not None else None),
                            "snapshot": snapshot})
            await asyncio.sleep(0.05)
        report["loop_completed"] = True
    except Exception as exc:  # noqa: BLE001 - retain evidence and always reap own demo threads
        report["error"] = str(exc)
    finally:
        await app.on_shutdown(None)
        report["final_command"] = app.simulation.snapshot()["command"]
        report["samples"] = samples
        report["observation_timings"] = list(app.observation_timings)
        report["continuity"] = continuity_summary(report["observation_timings"])
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "observation_timings")}, indent=2))
    return 0 if report.get("loop_completed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vln-server", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--instruction", default="go to the television")
    args = parser.parse_args()
    url = urlparse(args.vln_server)
    if url.scheme != "ws" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("explicit loopback ws URL required")
    if not 1 <= args.seconds <= 60 or not args.instruction.strip():
        parser.error("seconds must be 1..60 and instruction nonempty")
    if not args.output_dir.is_absolute() or args.output_dir.exists():
        parser.error("output-dir must be fresh and absolute")
    args.output_dir.mkdir(parents=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
