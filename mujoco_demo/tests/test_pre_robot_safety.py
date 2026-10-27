"""Pre-robot acceptance probes: simulation only, no hardware transport.

Freshness regressions must fail the acceptance gate, not be marked expected failures.
The deterministic inference/controller double isolates the demo orchestration;
separate tests exercise the actual external Soma DWA process.
"""
import asyncio
import contextlib
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vln_mujoco import server
from vln_mujoco.mpc import project_body_to_world
from vln_mujoco.robots.turtlebot import TurtleBotBackend
from vln_mujoco.simulation import COMMAND_TIMEOUT_S, Simulation
from vln_mujoco.vln_client import InferenceResult


class HeldPathTracker:
    """Deterministic planner double; not an MPC/DWA performance test."""

    def __init__(self):
        self.paths = []
        self.reset()

    def reset(self):
        self.command = (0.0, 0.0)
        self.error = ""

    def set_body_path(self, points, pose):
        self.paths.append(project_body_to_world(points, pose))
        self.command = (0.2, 0.0)
        self.error = ""

    def poll(self):
        return self.command

    def submit(self, pose):
        pass


class QueuedClient:
    def __init__(self, *args, **kwargs):
        self.results = []

    def offer_frame(self, *args):
        return False

    def take_results(self):
        results, self.results = self.results, []
        return results

    def stop(self):
        self.results.clear()


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setattr(server, "MpcTracker", HeldPathTracker)
    monkeypatch.setattr(server, "VlnClient", QueuedClient)
    app = server.VlnMujocoServer(simulation=Simulation(TurtleBotBackend()))
    # Avoid UI serialization: exercise the real update loop and simulation only.
    app.runtime_signature = lambda: None
    app.auto_owner = object()
    app.auto_mode = "vln"
    return app


def result(stamp, *, stop=False):
    return InferenceResult(1, stamp, ((1.0, 0.0, 0.0),) * 10,
                           stop, True, None, None, None, None, 0.0)


def capture(app, stamp, pose=(6.5, 13.8, 0), age=0):
    app.capture_poses[stamp] = pose
    app.capture_deadlines[stamp] = time.monotonic() + server.VLN_OBSERVATION_MAX_AGE_S - age


async def pump(app, duration):
    """Advance actual MuJoCo state while the real controller loop awaits replies."""
    task = asyncio.create_task(app.update_loop())
    deadline = time.monotonic() + duration
    try:
        while time.monotonic() < deadline:
            app.simulation._advance(time.monotonic())
            await asyncio.sleep(0.005)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("yaw", [0, math.pi / 2, -math.pi / 2, math.pi])
def test_camera_body_world_axes_and_capture_pose(yaw):
    robot = TurtleBotBackend()
    robot._set_planar_pose(6.5, 13.8, yaw)
    import mujoco

    mujoco.mj_forward(robot.model, robot.data)
    camera = robot.model.camera(robot.camera_name).id
    forward = -robot.data.cam_xmat[camera].reshape(3, 3)[:, 2]
    assert forward == pytest.approx([math.cos(yaw), math.sin(yaw), 0])
    capture = (6.5, 13.8, yaw)
    world = project_body_to_world([(1, 0, 0), (0, 1, 0)], capture)
    assert world[0, :2] == pytest.approx(np.array(capture[:2]) + forward[:2])
    assert world[1, :2] == pytest.approx([6.5 - math.sin(yaw), 13.8 + math.cos(yaw)])
    # Moving after capture must not alter that frozen transform.
    robot._set_planar_pose(8, 15, yaw + 0.5)
    assert project_body_to_world([(1, 0, 0)], capture)[0] == pytest.approx(world[0])


def test_command_watchdog_stops_when_publisher_disappears():
    sim = Simulation(TurtleBotBackend())
    sim.set_velocity(0.2, 0)
    sent = sim._command_at
    sim._advance(sent + 0.1)
    assert sim.robot.state().velocity[0] > 0
    sim._advance(sent + COMMAND_TIMEOUT_S + 0.001)
    assert sim.robot.state().velocity == pytest.approx((0, 0))


@pytest.mark.asyncio
async def test_missing_capture_pose_fails_closed(demo):
    demo.vln.results.append(result(time.time_ns()))
    await pump(demo, 0.05)
    assert demo.mpc.paths == []
    assert demo.simulation.robot.state().velocity == pytest.approx((0, 0))
    assert "missing robot pose" in demo.mpc.error


@pytest.mark.asyncio
async def test_delayed_reply_uses_capture_not_current_pose(demo):
    stamp = time.time_ns()
    capture(demo, stamp, (6.5, 13.8, math.pi / 2))
    demo.simulation.set_velocity(0.2, 0)
    initial_x = demo.simulation.robot.state().pose[0]
    # Physics advances during the simulated inference wait; no model output yet.
    await pump(demo, 0.12)
    assert demo.simulation.robot.state().pose[0] > initial_x
    demo.vln.results.append(result(stamp))
    await pump(demo, 0.05)
    assert demo.mpc.paths[0][0] == pytest.approx((6.5, 14.8, math.pi / 2))


@pytest.mark.asyncio
async def test_cancel_clears_path_and_rejects_late_reply_while_idle(demo):
    stamp = time.time_ns()
    capture(demo, stamp)
    demo.vln.results.append(result(stamp))
    await pump(demo, 0.1)
    demo.stop_vln()
    before = len(demo.mpc.paths)
    demo.vln.results.append(result(stamp))
    await pump(demo, 0.1)
    assert len(demo.mpc.paths) == before
    assert demo.simulation.robot.state().velocity == pytest.approx((0, 0))


@pytest.mark.asyncio
async def test_model_stop_does_not_mark_point_goal_reached(demo):
    stamp = time.time_ns()
    capture(demo, stamp)
    demo.vln.results.append(result(stamp, stop=True))
    await pump(demo, 0.05)
    assert demo.point_goal.state != "REACHED"
    assert demo.auto_owner is None
    assert demo.simulation.robot.state().velocity == pytest.approx((0, 0))


@pytest.mark.asyncio
async def test_expired_model_reply_must_not_be_adopted(demo):
    stamp = time.time_ns() - 2_000_000_000
    capture(demo, stamp, age=2)
    demo.vln.results.append(result(stamp))
    await pump(demo, 0.1)
    assert demo.mpc.paths == []


@pytest.mark.asyncio
async def test_camera_or_model_stall_must_expire_held_path(demo):
    stamp = time.time_ns()
    capture(demo, stamp)
    demo.vln.results.append(result(stamp))
    await pump(demo, 0.8)
    assert demo.simulation.robot.state().velocity == pytest.approx((0, 0))


@pytest.mark.asyncio
async def test_fresh_reply_recovers_after_expiry(demo):
    old = time.time_ns()
    capture(demo, old, age=2)
    demo.vln.results.append(result(old))
    await pump(demo, 0.05)
    assert demo.mpc.error
    fresh = time.time_ns()
    capture(demo, fresh)
    demo.vln.results.append(result(fresh))
    await pump(demo, 0.1)
    assert demo.simulation.robot.state().velocity[0] > 0
    assert not demo.mpc.error


@pytest.mark.asyncio
async def test_old_stop_does_not_end_current_task(demo):
    old = time.time_ns()
    capture(demo, old, age=2)
    demo.vln.results.append(result(old, stop=True))
    await pump(demo, 0.05)
    assert demo.auto_owner is not None
    assert demo.auto_mode == "vln"
    assert demo.simulation.robot.state().velocity == pytest.approx((0, 0))


@pytest.mark.asyncio
async def test_unclocked_capture_fails_closed(demo):
    stamp = time.time_ns()
    demo.capture_poses[stamp] = (6.5, 13.8, 0)
    demo.vln.results.append(result(stamp))
    await pump(demo, 0.05)
    assert demo.mpc.paths == []
    assert "unclocked" in demo.mpc.error


def test_invalid_model_numbers_rejected():
    from vln_mujoco.vln_client import VlnClient

    with pytest.raises(RuntimeError, match="non-finite"):
        VlnClient._parse_waypoints([[float("nan"), 0, 0]])


def test_late_mpc_solve_after_cancel_cannot_restore_command():
    from concurrent.futures import Future

    from vln_mujoco.mpc import MpcTracker

    # Test actual generation-check code, without constructing a numeric solver.
    tracker = MpcTracker.__new__(MpcTracker)
    tracker._generation = 1
    tracker.reset()
    tracker._future = Future()
    tracker._future.set_result(SimpleNamespace(generation=1, command=(0.4, 0)))
    assert tracker.poll() is None
    assert tracker.command == (0, 0)
