import math
from pathlib import Path

import pytest
from vln_mujoco.avoidance import LocalCollisionRecovery, wrap_angle
from vln_mujoco.robots.turtlebot import TurtleBotBackend


def finite_wall_scene(tmp_path: Path) -> Path:
    source = tmp_path / "finite-wall.xml"
    source.write_text(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="20 20 0.1" />
            <geom name="wall" type="box" pos="6.75 13.8 0.5" size="0.02 0.15 0.5" />
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return source


def test_collision_recovery_builds_a_bounded_detour() -> None:
    recovery = LocalCollisionRecovery()

    command = recovery.update(
        (0.5, 0.0),
        (0.0, 0.0, 0.0),
        collision_blocked=True,
    )
    assert command[0] < 0.0
    assert command[1] == 0.0
    assert recovery.snapshot()["phase"] == "backoff"

    command = recovery.update(
        (0.5, 0.0),
        (-0.13, 0.0, 0.0),
        collision_blocked=False,
    )
    assert command[0] == 0.0
    assert command[1] > 0.0
    assert recovery.snapshot()["phase"] == "turn_away"

    command = recovery.update(
        (0.5, 0.0),
        (-0.13, 0.0, math.pi / 2.0),
        collision_blocked=False,
    )
    assert command == pytest.approx((0.3, 0.0))
    assert recovery.snapshot()["phase"] == "bypass"

    command = recovery.update(
        (0.5, 0.0),
        (-0.13, 0.46, math.pi / 2.0),
        collision_blocked=False,
    )
    assert command[0] == 0.0
    assert command[1] < 0.0
    assert recovery.snapshot()["phase"] == "turn_forward"

    command = recovery.update(
        (0.5, 0.0),
        (-0.13, 0.46, 0.0),
        collision_blocked=False,
    )
    assert command == pytest.approx((0.3, 0.0))
    assert recovery.snapshot()["phase"] == "probe"

    command = recovery.update(
        (0.5, -0.2),
        (0.18, 0.46, 0.0),
        collision_blocked=False,
    )
    assert command == pytest.approx((0.5, -0.2))
    assert recovery.snapshot()["phase"] == "idle"
    assert recovery.snapshot()["events"] == 1


def test_collision_during_bypass_switches_side() -> None:
    recovery = LocalCollisionRecovery()
    recovery.update((0.5, 0.0), (0.0, 0.0, 0.0), collision_blocked=True)
    recovery.update(
        (0.5, 0.0),
        (-0.13, 0.0, 0.0),
        collision_blocked=False,
    )
    recovery.update(
        (0.5, 0.0),
        (-0.13, 0.0, math.pi / 2.0),
        collision_blocked=False,
    )

    command = recovery.update(
        (0.5, 0.0),
        (-0.13, 0.0, math.pi / 2.0),
        collision_blocked=True,
    )

    assert command[0] < 0.0
    assert command[1] == 0.0
    assert recovery.snapshot()["phase"] == "backoff"

    command = recovery.update(
        (0.5, 0.0),
        (-0.13, -0.13, math.pi / 2.0),
        collision_blocked=False,
    )
    assert command[0] == 0.0
    assert command[1] < 0.0
    assert recovery.snapshot()["direction"] == "right"
    assert recovery.snapshot()["attempts"] == 2


def test_idle_recovery_does_not_move_for_a_stopped_command() -> None:
    recovery = LocalCollisionRecovery()

    assert recovery.update(
        (0.0, 0.0),
        (1.0, 2.0, 0.0),
        collision_blocked=True,
    ) == (0.0, 0.0)
    assert recovery.snapshot()["phase"] == "idle"


def test_collision_recovery_drives_around_a_finite_wall(tmp_path: Path) -> None:
    robot = TurtleBotBackend(finite_wall_scene(tmp_path))
    recovery = LocalCollisionRecovery()
    goal = (7.4, 13.8)
    effective = (0.0, 0.0)
    phases: set[str] = set()

    for step in range(6000):
        if step % 20 == 0:
            x, y, _, yaw = robot.state().pose
            bearing = wrap_angle(math.atan2(goal[1] - y, goal[0] - x) - yaw)
            desired = (
                0.5 * max(0.0, math.cos(bearing)),
                max(-1.0, min(1.0, 2.0 * bearing)),
            )
            guard = robot.state().telemetry["collision_guard"]
            effective = recovery.update(
                desired,
                (x, y, yaw),
                collision_blocked=bool(guard["blocked"]),
            )
            phases.add(recovery.phase)
        robot.step(effective)
        assert robot._max_penetration(robot._obstacle_contacts()) == 0.0
        x, y, _, _ = robot.state().pose
        if math.hypot(goal[0] - x, goal[1] - y) < 0.15:
            break

    x, y, _, _ = robot.state().pose
    assert math.hypot(goal[0] - x, goal[1] - y) < 0.15
    assert {"turn_away", "bypass", "turn_forward", "probe"} <= phases
    assert recovery.snapshot()["events"] >= 1
    assert recovery.snapshot()["phase"] != "failed"
