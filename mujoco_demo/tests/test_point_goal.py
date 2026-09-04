import math

import pytest
from vln_mujoco.grid_planner import GridPlan
from vln_mujoco.point_goal import PointGoalState, straight_body_path


def test_straight_body_path_reaches_requested_point() -> None:
    path = straight_body_path(0.3, 0.4, spacing_m=0.1)

    assert len(path) == 5
    assert path[-1] == pytest.approx((0.3, 0.4, math.atan2(0.4, 0.3)))


def test_point_goal_is_fixed_in_world_after_start() -> None:
    goal = PointGoalState(tolerance_m=0.15)
    path = goal.start((2.0, 3.0, math.pi / 2.0), 1.0, 0.5)

    assert path
    assert goal.snapshot((2.0, 3.0, math.pi / 2.0))["goal_world"] == pytest.approx(
        {"x": 1.5, "y": 4.0}
    )
    snapshot = goal.snapshot((1.5, 3.5, math.pi / 2.0))
    assert snapshot["relative_goal"] == pytest.approx({"forward": 0.5, "left": 0.0})
    assert snapshot["distance_m"] == pytest.approx(math.hypot(1.0, 0.5))


def test_point_goal_has_independent_geometric_arrival_check() -> None:
    goal = PointGoalState(tolerance_m=0.15)
    goal.start((0.0, 0.0, 0.0), 1.0, 0.0)

    assert goal.update((0.8, 0.0, 1.0)) is False
    assert goal.state == "RUNNING"
    assert goal.update((0.86, 0.0, -2.0)) is True
    assert goal.state == "REACHED"
    assert goal.active is False
    assert goal.snapshot()["distance_m"] == pytest.approx(0.14)


def test_point_goal_exposes_installed_astar_plan_in_current_body_frame() -> None:
    goal = PointGoalState(tolerance_m=0.15)
    goal.start((1.0, 2.0, 0.0), 1.0, 0.0, planner="astar")
    assert goal.state == "PLANNING"
    plan = GridPlan(
        world_path=((1.0, 2.0, 0.0), (1.5, 2.5, 0.0), (2.0, 2.0, 0.0)),
        raw_cells=((0, 0), (1, 1), (0, 2)),
        simplified_cells=((0, 0), (1, 1), (0, 2)),
        planning_ms=12.5,
        grid_resolution_m=0.1,
        grid_width=20,
        grid_height=10,
        blocked_cells=4,
    )

    goal.install_plan(plan)
    snapshot = goal.snapshot((1.0, 2.0, 0.0))

    assert goal.state == "RUNNING"
    assert snapshot["planner"] == "astar"
    assert snapshot["planning_ms"] == pytest.approx(12.5)
    assert snapshot["raw_path_cells"] == 3
    assert snapshot["grid"] == {
        "resolution_m": 0.1,
        "width": 20,
        "height": 10,
        "blocked_cells": 4,
    }
    assert snapshot["waypoints"][-1][:2] == pytest.approx((1.0, 0.0))


@pytest.mark.parametrize(
    ("forward", "left", "message"),
    [
        (float("nan"), 0.0, "finite"),
        (0.1, 0.0, "farther"),
        (26.0, 0.0, "25 m"),
    ],
)
def test_point_goal_rejects_invalid_targets(
    forward: float,
    left: float,
    message: str,
) -> None:
    goal = PointGoalState(tolerance_m=0.15)

    with pytest.raises(ValueError, match=message):
        goal.start((0.0, 0.0, 0.0), forward, left)
