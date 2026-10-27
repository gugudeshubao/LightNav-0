import math
from pathlib import Path

import pytest
from vln_mujoco.grid_planner import (
    OccupancyGrid,
    astar_path,
    has_line_of_sight,
    sample_world_polyline,
    simplify_cell_path,
)
from vln_mujoco.robots.turtlebot import TurtleBotBackend


def grid(
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> OccupancyGrid:
    return OccupancyGrid(0.0, 0.0, 1.0, width, height, frozenset(blocked))


def finite_wall_scene(tmp_path: Path) -> Path:
    source = tmp_path / "finite-wall.xml"
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


def test_astar_routes_around_wall_and_simplifies_visible_segments() -> None:
    occupancy = grid(7, 7, {(row, 3) for row in range(1, 6)})

    path = astar_path(occupancy, (3, 1), (3, 5))
    simplified = simplify_cell_path(occupancy, path)

    assert path[0] == (3, 1)
    assert path[-1] == (3, 5)
    assert all(occupancy.is_free(cell) for cell in path)
    assert len(simplified) < len(path)
    assert all(
        has_line_of_sight(occupancy, start, goal)
        for start, goal in zip(simplified, simplified[1:])
    )


def test_astar_does_not_cut_a_blocked_diagonal_corner() -> None:
    occupancy = grid(2, 2, {(0, 1), (1, 0)})

    with pytest.raises(ValueError, match="no collision-free path"):
        astar_path(occupancy, (0, 0), (1, 1))


def test_sample_world_polyline_has_dense_tangent_aligned_points() -> None:
    path = sample_world_polyline(
        ((0.0, 0.0), (0.25, 0.0), (0.25, 0.2)),
        spacing_m=0.1,
    )

    assert path[-1] == pytest.approx((0.25, 0.2, math.pi / 2.0))
    assert max(
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(path, path[1:])
    ) <= 0.1 + 1e-9


def test_turtlebot_map_planner_routes_around_finite_wall(tmp_path: Path) -> None:
    robot = TurtleBotBackend(finite_wall_scene(tmp_path))

    plan = robot.plan_world_path(
        (6.5, 13.8, 0.0),
        (7.0, 13.8),
        resolution_m=0.1,
        padding_m=1.5,
    )

    assert plan.world_path[0][:2] == pytest.approx((6.5, 13.8))
    assert plan.world_path[-1][:2] == pytest.approx((7.0, 13.8))
    assert any(abs(point[1] - 13.8) > 1.1 for point in plan.world_path)
    assert plan.path_length_m > 2.0
    assert len(plan.simplified_cells) < len(plan.raw_cells)
    assert plan.blocked_cells > 0
