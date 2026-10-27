"""PointGoal state and straight-line reference generation for the MuJoCo demo."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from .grid_planner import GridPlan

GOAL_TOLERANCE_M = 0.15
PATH_SPACING_M = 0.10
MAX_GOAL_DISTANCE_M = 25.0


def _pose3(values: Sequence[float]) -> tuple[float, float, float]:
    pose = tuple(float(value) for value in values)
    if len(pose) != 3 or not all(math.isfinite(value) for value in pose):
        raise ValueError("pose must be a finite 3-vector")
    return pose


def straight_body_path(
    forward_m: float,
    left_m: float,
    *,
    spacing_m: float = PATH_SPACING_M,
) -> tuple[tuple[float, float, float], ...]:
    """Sample a straight body-frame path to an unconstrained-heading point goal."""
    forward = float(forward_m)
    left = float(left_m)
    spacing = float(spacing_m)
    if not all(math.isfinite(value) for value in (forward, left, spacing)):
        raise ValueError("PointGoal values must be finite")
    if spacing <= 0.0:
        raise ValueError("PointGoal path spacing must be positive")
    distance = math.hypot(forward, left)
    if distance <= 0.0:
        return ()
    if distance > MAX_GOAL_DISTANCE_M:
        raise ValueError(f"PointGoal distance must not exceed {MAX_GOAL_DISTANCE_M:g} m")
    count = max(1, math.ceil(distance / spacing))
    bearing = math.atan2(left, forward)
    return tuple(
        (forward * index / count, left * index / count, bearing)
        for index in range(1, count + 1)
    )


class PointGoalState:
    """Track one fixed world-frame goal derived from a start-relative target."""

    def __init__(self, *, tolerance_m: float = GOAL_TOLERANCE_M) -> None:
        self.tolerance_m = float(tolerance_m)
        if not math.isfinite(self.tolerance_m) or self.tolerance_m <= 0.0:
            raise ValueError("PointGoal tolerance must be positive")
        self.reset()

    @property
    def active(self) -> bool:
        return self.state == "RUNNING"

    def reset(self) -> None:
        self.state = "IDLE"
        self.requested_body: tuple[float, float] | None = None
        self.goal_world: tuple[float, float] | None = None
        self.start_pose: tuple[float, float, float] | None = None
        self.distance_m: float | None = None
        self.started_s: float | None = None
        self.elapsed_s: float | None = None
        self.planner = "direct"
        self.planning_ms: float | None = None
        self.grid: dict[str, int | float] | None = None
        self.raw_path_cells: int | None = None
        self.simplified_path_cells: int | None = None
        self.path_length_m: float | None = None
        self.world_path: tuple[tuple[float, float, float], ...] = ()
        self.error = ""

    def start(
        self,
        start_pose: Sequence[float],
        forward_m: float,
        left_m: float,
        *,
        planner: str = "direct",
    ) -> tuple[tuple[float, float, float], ...]:
        pose = _pose3(start_pose)
        planner_name = str(planner)
        if planner_name not in {"direct", "astar"}:
            raise ValueError("PointGoal planner must be 'direct' or 'astar'")
        path = straight_body_path(forward_m, left_m)
        distance = math.hypot(float(forward_m), float(left_m))
        if distance <= self.tolerance_m:
            raise ValueError(
                f"PointGoal must be farther than the {self.tolerance_m:g} m goal tolerance"
            )
        cosine = math.cos(pose[2])
        sine = math.sin(pose[2])
        forward = float(forward_m)
        left = float(left_m)
        self.requested_body = (forward, left)
        self.goal_world = (
            pose[0] + cosine * forward - sine * left,
            pose[1] + sine * forward + cosine * left,
        )
        self.start_pose = pose
        self.distance_m = distance
        self.started_s = time.monotonic()
        self.elapsed_s = 0.0
        self.planner = planner_name
        self.planning_ms = 0.0 if self.planner == "direct" else None
        self.grid = None
        self.raw_path_cells = None
        self.simplified_path_cells = None
        self.path_length_m = distance
        if self.planner == "direct":
            bearing_world = pose[2] + math.atan2(left, forward)
            cosine = math.cos(pose[2])
            sine = math.sin(pose[2])
            projected = tuple(
                (
                    pose[0] + cosine * point[0] - sine * point[1],
                    pose[1] + sine * point[0] + cosine * point[1],
                    bearing_world,
                )
                for point in path
            )
            self.world_path = ((pose[0], pose[1], bearing_world), *projected)
        else:
            self.world_path = ()
        self.error = ""
        self.state = "RUNNING" if self.planner == "direct" else "PLANNING"
        return path

    def install_plan(self, plan: GridPlan) -> None:
        if self.planner != "astar" or self.state != "PLANNING":
            raise RuntimeError("PointGoal is not waiting for an A* plan")
        self.world_path = plan.world_path
        self.planning_ms = plan.planning_ms
        self.grid = {
            "resolution_m": plan.grid_resolution_m,
            "width": plan.grid_width,
            "height": plan.grid_height,
            "blocked_cells": plan.blocked_cells,
        }
        self.raw_path_cells = len(plan.raw_cells)
        self.simplified_path_cells = len(plan.simplified_cells)
        self.path_length_m = plan.path_length_m
        self.state = "RUNNING"

    def update(self, pose: Sequence[float]) -> bool:
        """Refresh distance and return whether the independent goal test passed."""
        current = _pose3(pose)
        if self.goal_world is None:
            return False
        self.distance_m = math.hypot(
            self.goal_world[0] - current[0],
            self.goal_world[1] - current[1],
        )
        if self.started_s is not None:
            self.elapsed_s = time.monotonic() - self.started_s
        if self.active and self.distance_m <= self.tolerance_m:
            self.state = "REACHED"
            return True
        return False

    def stop(self) -> None:
        if self.state in {"PLANNING", "RUNNING"}:
            self.state = "STOPPED"
            if self.started_s is not None:
                self.elapsed_s = time.monotonic() - self.started_s

    def fail(self, message: str) -> None:
        self.error = str(message)
        self.state = "FAILED"
        if self.started_s is not None:
            self.elapsed_s = time.monotonic() - self.started_s

    def _relative_goal(
        self,
        pose: Sequence[float] | None,
    ) -> tuple[float, float] | None:
        if pose is None or self.goal_world is None:
            return None
        current = _pose3(pose)
        delta_x = self.goal_world[0] - current[0]
        delta_y = self.goal_world[1] - current[1]
        cosine = math.cos(current[2])
        sine = math.sin(current[2])
        return (
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
        )

    def snapshot(self, pose: Sequence[float] | None = None) -> dict[str, object]:
        relative = self._relative_goal(pose)
        waypoints: tuple[tuple[float, float, float], ...] = ()
        if pose is not None and self.world_path:
            current = _pose3(pose)
            nearest = min(
                range(len(self.world_path)),
                key=lambda index: math.hypot(
                    self.world_path[index][0] - current[0],
                    self.world_path[index][1] - current[1],
                ),
            )
            cosine = math.cos(current[2])
            sine = math.sin(current[2])
            waypoints = tuple(
                (
                    cosine * (point[0] - current[0])
                    + sine * (point[1] - current[1]),
                    -sine * (point[0] - current[0])
                    + cosine * (point[1] - current[1]),
                    math.atan2(
                        math.sin(point[2] - current[2]),
                        math.cos(point[2] - current[2]),
                    ),
                )
                for point in self.world_path[nearest:]
            )
        return {
            "state": self.state,
            "active": self.active,
            "requested_body": (
                None
                if self.requested_body is None
                else {"forward": self.requested_body[0], "left": self.requested_body[1]}
            ),
            "goal_world": (
                None
                if self.goal_world is None
                else {"x": self.goal_world[0], "y": self.goal_world[1]}
            ),
            "relative_goal": (
                None
                if relative is None
                else {"forward": relative[0], "left": relative[1]}
            ),
            "distance_m": self.distance_m,
            "tolerance_m": self.tolerance_m,
            "elapsed_s": self.elapsed_s,
            "planner": self.planner,
            "planning_ms": self.planning_ms,
            "grid": self.grid,
            "raw_path_cells": self.raw_path_cells,
            "simplified_path_cells": self.simplified_path_cells,
            "path_length_m": self.path_length_m,
            "waypoints": [list(point) for point in waypoints],
            "error": self.error,
        }
