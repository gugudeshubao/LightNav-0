"""Capture deterministic, open-loop simulation observations; never actuate hardware."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route", choices=("left", "right", "forward"), required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--instruction", default="go to the television")
    args = parser.parse_args()
    if not 1 <= args.frames <= 256 or not args.instruction.strip():
        parser.error("frames must be in [1,256] and instruction nonempty")
    if not args.output_dir.is_absolute() or args.output_dir.exists():
        parser.error("output-dir must be a fresh absolute directory")

    import mujoco
    from PIL import Image
    from vln_mujoco.model import PHYSICS_DT
    from vln_mujoco.robots.turtlebot import TurtleBotBackend

    robot = TurtleBotBackend()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    observations = []
    steps = round(0.25 / PHYSICS_DT)
    assert abs(steps * PHYSICS_DT - 0.25) < 1e-9
    with mujoco.Renderer(robot.model, height=270, width=480) as renderer:
        for index in range(args.frames):
            command = (0.20, 0.0) if args.route == "forward" else (
                0.0 if index < 8 else 0.15,
                (0.45 if args.route == "left" else -0.45) if index < 8 else 0.0,
            )
            if index:
                for _ in range(steps):
                    robot.step(command)
            renderer.update_scene(robot.data, camera=robot.camera_name)
            rgb = renderer.render()
            filename = f"frame-{index:04d}.png"
            Image.fromarray(rgb).save(args.output_dir / filename)
            state = robot.state()
            packed_path = args.output_dir / f"frame-{index:04d}.rgb"
            packed_path.write_bytes(rgb.tobytes())
            probe_input = {
                "schema": "soma-lightnav-simulation-input/v1", "actuation": False,
                "width": 480, "height": 270, "rgb_path": str(packed_path),
                "capture_pose_xyz_yaw": state.pose,
                "source_stamp_ns": 1 + index * 250_000_000,
            }
            (args.output_dir / f"soma-input-{index:04d}.json").write_text(
                json.dumps(probe_input, indent=2) + "\n")
            observations.append({
                "file": filename, "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "simulation_time_s": float(robot.data.time),
                "capture_stamp_ns": 1 + index * 250_000_000,
                "pose_xyz_yaw": state.pose, "velocity": state.velocity,
                "command": command if index else (0.0, 0.0),
                "telemetry": state.telemetry,
            })
    manifest = {
        "schema": "lightnav-simulation-capture/v1", "actuation": False,
        "control": "scripted open-loop; not model or DWA controlled",
        "scene": "bundled procthor-10k-val/val_2_ceiling.xml",
        "independent_scene": False, "route": args.route,
        "instruction": args.instruction, "frame_size": [480, 270],
        "timeline": "simulation clock; synchronous RGB and ground-truth pose at 4Hz",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("mujoco", "numpy", "pillow")},
        "observations": observations,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"frames": len(observations), "route": args.route,
                      "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
