import time
from pathlib import Path

import pytest
from vln_mujoco.robots.base import RobotState
from vln_mujoco.robots.turtlebot import TurtleBotBackend
from vln_mujoco.simulation import COMMAND_TIMEOUT_S, CameraFrame, Simulation


class TelemetryTurtleBot(TurtleBotBackend):
    def state(self) -> RobotState:
        state = super().state()
        return RobotState(
            pose=state.pose,
            velocity=state.velocity,
            telemetry={"pose": "backend-pose", "policy_command": {"linear": 0.3}},
        )


def obstacle_scene(tmp_path: Path) -> Path:
    source = tmp_path / "obstacle.xml"
    source.write_text(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="20 20 0.1" />
            <geom name="wall" type="box" pos="6.75 13.8 0.5" size="0.02 1 0.5" />
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return source


def test_turtlebot_backend_integrates_velocity_command() -> None:
    robot = TurtleBotBackend()
    start = robot.state()

    for _ in range(20):
        robot.step((0.4, 0.2))

    state = robot.state()
    assert state.pose[0] > start.pose[0]
    assert state.pose[1] > start.pose[1]
    assert state.pose[3] > start.pose[3]
    assert state.velocity == pytest.approx((0.4, 0.2))


def test_turtlebot_backend_reset_restores_spawn_pose() -> None:
    robot = TurtleBotBackend()
    robot.step((0.4, 0.2))

    robot.reset()

    state = robot.state()
    assert state.pose == pytest.approx((6.5, 13.8, 0.033, 0.0))
    assert state.velocity == pytest.approx((0.0, 0.0))


def test_turtlebot_collision_guard_blocks_wall_but_allows_turn(
    tmp_path: Path,
) -> None:
    robot = TurtleBotBackend(obstacle_scene(tmp_path))

    for _ in range(100):
        robot.step((1.0, 0.0))

    blocked = robot.state()
    assert blocked.pose[0] == pytest.approx(6.635)
    assert blocked.pose[1] == pytest.approx(13.8)
    assert blocked.velocity == pytest.approx((0.0, 0.0))
    assert blocked.telemetry["collision_guard"] == {
        "blocked": True,
        "events": 1,
        "contacts": ["base_collision:wall"],
        "attempted_penetration_m": pytest.approx(0.002),
    }

    for _ in range(20):
        robot.step((1.0, 0.0))
    assert robot.state().pose == pytest.approx(blocked.pose)
    assert robot.state().telemetry["collision_guard"]["events"] == 1

    robot.step((0.0, 1.0))
    turning = robot.state()
    assert turning.pose[0:3] == pytest.approx(blocked.pose[0:3])
    assert turning.pose[3] > blocked.pose[3]
    assert turning.velocity == pytest.approx((0.0, 1.0))
    assert turning.telemetry["collision_guard"]["blocked"] is False


def test_turtlebot_reset_clears_collision_telemetry(tmp_path: Path) -> None:
    robot = TurtleBotBackend(obstacle_scene(tmp_path))
    for _ in range(100):
        robot.step((1.0, 0.0))
    assert robot.state().telemetry["collision_guard"]["blocked"] is True

    robot.reset()

    assert robot.state().telemetry["collision_guard"] == {
        "blocked": False,
        "events": 0,
        "contacts": [],
        "attempted_penetration_m": 0.0,
    }


def test_simulation_owns_command_watchdog_and_robot_identity() -> None:
    simulation = Simulation(TurtleBotBackend())
    simulation.set_velocity(0.4, 0.2)
    simulation._advance(time.monotonic())

    assert simulation.robot_name == "TurtleBot"
    assert simulation.snapshot()["command"] == pytest.approx(
        {"linear": 0.4, "angular": 0.2}
    )

    simulation._command_at = time.monotonic() - COMMAND_TIMEOUT_S - 0.1
    simulation._advance(time.monotonic())

    assert simulation.snapshot()["command"] == pytest.approx(
        {"linear": 0.0, "angular": 0.0}
    )


def test_simulation_namespaces_backend_telemetry() -> None:
    simulation = Simulation(TelemetryTurtleBot())

    snapshot = simulation.snapshot()

    assert isinstance(snapshot["pose"], dict)
    assert snapshot["backend"] == {
        "pose": "backend-pose",
        "policy_command": {"linear": 0.3},
    }


def test_simulation_reset_discards_pre_reset_camera_frames() -> None:
    simulation = Simulation(TurtleBotBackend())
    stale = CameraFrame(123, b"rgb", b"jpeg", (9.0, 8.0, 0.7))
    simulation._frame = stale
    simulation._third_person_frame = stale
    simulation._frame_count = 4

    simulation.reset()

    assert simulation.frame() is None
    assert simulation.third_person_frame() is None
    assert simulation.snapshot()["camera"]["ready"] is False
    assert simulation._frame_count == 0
