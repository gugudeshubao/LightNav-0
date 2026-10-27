"""Actual Soma DWA binary + MuJoCo; enable with SOMA_DWA_PLANNER.

Synthetic model paths and known obstacle samples isolate controller behaviour;
these tests do not verify model semantics or real sensor obstacle extraction.
"""
import json
import math
import os
import select
import subprocess
from pathlib import Path

import pytest

from vln_mujoco.robots.turtlebot import TurtleBotBackend


@pytest.fixture
def planner():
    binary = os.environ.get("SOMA_DWA_PLANNER")
    if not binary:
        pytest.skip("requires explicit pure simulation Soma DWA binary")
    assert Path(binary).is_absolute() and Path(binary).is_file()
    process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, bufsize=1)

    def step(robot, capture, points, obstacles, count=10, stop=False):
        state = robot.state()
        x, y, _, yaw = state.pose
        v, w = state.velocity
        request = {"simulation_only": True, "capture_pose": capture,
                   "state": [x, y, yaw, v, 0, w], "local_trajectory": points,
                   "stop": stop, "point_count": count, "obstacles": obstacles}
        process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 5)[0], "DWA timeout"
        reply = json.loads(process.stdout.readline())
        assert reply["ok"] and reply["hardware_actuation"] is False
        assert all(math.isfinite(x) for x in reply["command"])
        return reply

    try:
        yield step
    finally:
        process.stdin.close()
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            pytest.fail("DWA failed to exit on EOF")
        assert code == 0


def scene(tmp_path):
    path = tmp_path / "wall.xml"
    path.write_text('<mujoco><worldbody><geom type="plane" size="20 20 .1"/>'
                    '<geom name="wall" type="box" pos="7.5 13.8 .5" '
                    'size=".02 2 .5"/></worldbody></mujoco>')
    return TurtleBotBackend(path)


@pytest.mark.parametrize("count", [10, 5])
def test_dwa_respects_wall_without_simulator_guard(planner, tmp_path, count, record_property):
    robot = scene(tmp_path)
    points = [{"x": 0.3 * (i + 1), "y": 0, "yaw": 0} for i in range(10)]
    obstacles = [[7.48, 11.8 + i * 0.025] for i in range(161)]
    xs = []
    clearances = []
    for _ in range(80):
        reply = planner(robot, [6.5, 13.8, 0], points, obstacles, count)
        for _ in range(10):
            robot.step((reply["command"][0], reply["command"][2]))
            x, _, _, yaw = robot.state().pose
            half_x, half_y = reply["footprint_half_extents_m"]
            projected_half_width = abs(math.cos(yaw)) * half_x + abs(math.sin(yaw)) * half_y
            clearances.append(7.48 - x - projected_half_width)
        xs.append(robot.state().pose[0])
    assert max(xs) > 6.5, "must demonstrate progress, not permanent zero"
    # A local planner may turn and follow a wall, so use its rotated footprint,
    # not a constant forward half-length or a requirement to remain stationary.
    record_property("minimum_rotated_footprint_clearance_m", min(clearances))
    record_property("maximum_center_x_m", max(xs))
    record_property("final_pose_xyz_yaw", robot.state().pose)
    assert min(clearances) >= -1e-6, f"minimum wall clearance: {min(clearances)}"
    assert robot.state().telemetry["collision_guard"]["events"] == 0


@pytest.mark.parametrize("count", [10, 5])
def test_sudden_obstacle_inside_footprint_stops_existing_motion(planner, tmp_path, count):
    robot = scene(tmp_path)
    points = [{"x": 0.3 * (i + 1), "y": 0, "yaw": 0} for i in range(10)]
    free = planner(robot, [6.5, 13.8, 0], points, [], count)
    assert free["command"][0] > 0
    for _ in range(10):
        robot.step((free["command"][0], free["command"][2]))
    x, y = robot.state().pose[:2]
    blocked = planner(robot, [6.5, 13.8, 0], points, [[x, y]], count)
    assert blocked["command"] == pytest.approx([0, 0, 0])
    robot.step((blocked["command"][0], blocked["command"][2]))
    assert robot.state().velocity == pytest.approx((0, 0))


def test_rotation_sweep_and_model_stop_are_not_arrival(planner, tmp_path):
    robot = scene(tmp_path)
    points = [{"x": 0, "y": 0, "yaw": 1.0} for _ in range(10)]
    free = planner(robot, [6.5, 13.8, 0], points, [])
    assert free["mode"] == "heading" and free["command"][2] > 0
    blocked = planner(robot, [6.5, 13.8, 0], points, [[6.5, 13.8]])
    assert blocked["command"] == pytest.approx([0, 0, 0])
    stop = planner(robot, [6.5, 13.8, 0],
                   [{"x": 0, "y": 0, "yaw": 0}] * 10, [], stop=True)
    assert stop["mode"] == "model_stop"
    assert stop["local_path_arrived"] is False
    assert stop["command"] == pytest.approx([0, 0, 0])
