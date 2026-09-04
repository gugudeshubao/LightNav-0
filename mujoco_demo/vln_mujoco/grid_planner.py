"""Small occupancy-grid planner used by the map-aware PointGoal demo."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

GridCell = tuple[int, int]
WorldPoint = tuple[float, float]


@dataclass(frozen=True)
class OccupancyGrid:
    """A world-aligned grid whose cells represent tested robot-centre poses."""

    origin_x: float
    origin_y: float
    resolution_m: float
    width: int
    height: int
    blocked: frozenset[GridCell]

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(value)
                for value in (self.origin_x, self.origin_y, self.resolution_m)
            )
            or self.resolution_m <= 0.0
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("invalid occupancy grid geometry")
        if any(not self.contains(cell) for cell in self.blocked):
            raise ValueError("blocked cell lies outside occupancy grid")

    def contains(self, cell: GridCell) -> bool:
        row, column = cell
        return 0 <= row < self.height and 0 <= column < self.width

    def is_free(self, cell: GridCell) -> bool:
        return self.contains(cell) and cell not in self.blocked

    def cell_to_world(self, cell: GridCell) -> WorldPoint:
        if not self.contains(cell):
            raise ValueError("cell lies outside occupancy grid")
        row, column = cell
        return (
            self.origin_x + column * self.resolution_m,
            self.origin_y + row * self.resolution_m,
        )

    def world_to_cell(self, point: Sequence[float]) -> GridCell:
        x, y = _point2(point)
        return (
            round((y - self.origin_y) / self.resolution_m),
            round((x - self.origin_x) / self.resolution_m),
        )


@dataclass(frozen=True)
class GridPlan:
    """A planned and controller-ready world-frame PointGoal route."""

    world_path: tuple[tuple[float, float, float], ...]
    raw_cells: tuple[GridCell, ...]
    simplified_cells: tuple[GridCell, ...]
    planning_ms: float
    grid_resolution_m: float
    grid_width: int
    grid_height: int
    blocked_cells: int

    @property
    def path_length_m(self) -> float:
        return sum(
            math.hypot(current[0] - previous[0], current[1] - previous[1])
            for previous, current in zip(self.world_path, self.world_path[1:])
        )


def _point2(values: Sequence[float]) -> WorldPoint:
    point = tuple(float(value) for value in values)
    if len(point) != 2 or not all(math.isfinite(value) for value in point):
        raise ValueError("point must be a finite 2-vector")
    return point


def _octile_distance(first: GridCell, second: GridCell) -> float:
    delta_row = abs(first[0] - second[0])
    delta_column = abs(first[1] - second[1])
    diagonal = min(delta_row, delta_column)
    return max(delta_row, delta_column) + (math.sqrt(2.0) - 1.0) * diagonal


def _neighbors(grid: OccupancyGrid, cell: GridCell) -> Iterable[tuple[GridCell, float]]:
    row, column = cell
    for delta_row in (-1, 0, 1):
        for delta_column in (-1, 0, 1):
            if delta_row == 0 and delta_column == 0:
                continue
            neighbor = (row + delta_row, column + delta_column)
            if not grid.is_free(neighbor):
                continue
            if delta_row and delta_column:
                # A diagonal is legal only when the robot centre could also
                # traverse both orthogonal cells. This prevents corner cuts.
                if not grid.is_free((row + delta_row, column)) or not grid.is_free(
                    (row, column + delta_column)
                ):
                    continue
                cost = math.sqrt(2.0)
            else:
                cost = 1.0
            yield neighbor, cost


def astar_path(
    grid: OccupancyGrid,
    start: GridCell,
    goal: GridCell,
) -> tuple[GridCell, ...]:
    """Return an eight-connected shortest path without diagonal corner cuts."""
    if not grid.contains(start) or not grid.contains(goal):
        raise ValueError("PointGoal lies outside the planning grid")
    if not grid.is_free(start):
        raise ValueError("PointGoal start pose is occupied")
    if not grid.is_free(goal):
        raise ValueError("PointGoal target pose is occupied")
    if start == goal:
        return (start,)

    frontier: list[tuple[float, float, GridCell]] = [(0.0, 0.0, start)]
    parents: dict[GridCell, GridCell] = {}
    costs = {start: 0.0}
    while frontier:
        _, popped_cost, current = heapq.heappop(frontier)
        if popped_cost > costs.get(current, math.inf):
            continue
        if current == goal:
            cells = [goal]
            while cells[-1] != start:
                cells.append(parents[cells[-1]])
            cells.reverse()
            return tuple(cells)
        for neighbor, step_cost in _neighbors(grid, current):
            candidate = popped_cost + step_cost
            if candidate >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = candidate
            parents[neighbor] = current
            priority = candidate + _octile_distance(neighbor, goal)
            heapq.heappush(frontier, (priority, candidate, neighbor))
    raise ValueError("no collision-free path to PointGoal")


def line_cells(start: GridCell, goal: GridCell) -> tuple[GridCell, ...]:
    """Return grid cells visited by a centre-to-centre line segment."""
    row, column = start
    goal_row, goal_column = goal
    row_count = abs(goal_row - row)
    column_count = abs(goal_column - column)
    row_step = 1 if goal_row > row else -1
    column_step = 1 if goal_column > column else -1
    row_index = 0
    column_index = 0
    cells = [(row, column)]
    while row_index < row_count or column_index < column_count:
        decision = (1 + 2 * column_index) * row_count - (
            1 + 2 * row_index
        ) * column_count
        if decision == 0:
            row += row_step
            column += column_step
            row_index += 1
            column_index += 1
        elif decision < 0:
            column += column_step
            column_index += 1
        else:
            row += row_step
            row_index += 1
        cells.append((row, column))
    return tuple(cells)


def has_line_of_sight(
    grid: OccupancyGrid,
    start: GridCell,
    goal: GridCell,
) -> bool:
    cells = line_cells(start, goal)
    if any(not grid.is_free(cell) for cell in cells):
        return False
    for previous, current in zip(cells, cells[1:]):
        delta_row = current[0] - previous[0]
        delta_column = current[1] - previous[1]
        if delta_row and delta_column:
            if not grid.is_free((previous[0] + delta_row, previous[1])):
                return False
            if not grid.is_free((previous[0], previous[1] + delta_column)):
                return False
    return True


def simplify_cell_path(
    grid: OccupancyGrid,
    path: Sequence[GridCell],
) -> tuple[GridCell, ...]:
    """Greedily remove A* cells while retaining collision-free line segments."""
    cells = tuple(path)
    if len(cells) <= 2:
        return cells
    simplified = [cells[0]]
    anchor = 0
    while anchor < len(cells) - 1:
        candidate = len(cells) - 1
        while candidate > anchor + 1 and not has_line_of_sight(
            grid,
            cells[anchor],
            cells[candidate],
        ):
            candidate -= 1
        simplified.append(cells[candidate])
        anchor = candidate
    return tuple(simplified)


def sample_world_polyline(
    points: Sequence[Sequence[float]],
    *,
    spacing_m: float,
) -> tuple[tuple[float, float, float], ...]:
    """Densify a world polyline and attach tangent yaw to each sample."""
    vertices = tuple(_point2(point) for point in points)
    spacing = float(spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("path spacing must be positive")
    if len(vertices) < 2:
        return ()
    sampled: list[tuple[float, float, float]] = []
    for start, goal in zip(vertices, vertices[1:]):
        delta_x = goal[0] - start[0]
        delta_y = goal[1] - start[1]
        distance = math.hypot(delta_x, delta_y)
        if distance <= 1e-12:
            continue
        count = max(1, math.ceil(distance / spacing))
        yaw = math.atan2(delta_y, delta_x)
        for index in range(1, count + 1):
            fraction = index / count
            sampled.append(
                (
                    start[0] + delta_x * fraction,
                    start[1] + delta_y * fraction,
                    yaw,
                )
            )
    return tuple(sampled)
