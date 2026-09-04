from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter
from typing import Sequence

import mujoco

from ..grid_planner import (
    GridPlan,
    OccupancyGrid,
    astar_path,
    sample_world_polyline,
    simplify_cell_path,
)
from ..model import PHYSICS_DT, SPAWN, compile_model, load_scene_xml
from .base import RenderCamera, RobotState, Twist2D

CAMERA_NAME = "robot_rgb"
THIRD_PERSON_CAMERA_NAME = "robot_third_person"
WHEEL_RADIUS = 0.033
WHEEL_TRACK = 0.160
COLLISION_EPS_M = 1e-6
OBSTACLE_NORMAL_XY_MIN = 0.5
PLANNER_RESOLUTION_M = 0.10
PLANNER_PADDING_M = 1.50
# The TurtleBot footprint is close to circular, but the wheels extend beyond
# the base cylinder. Testing eight headings over its pi-periodic footprint
# makes the occupancy grid conservative enough to allow in-place turns.
PLANNER_YAWS = tuple(index * math.pi / 8.0 for index in range(8))


def _robot_body() -> ET.Element:
    return ET.fromstring(
        f"""
        <body name="base_link" pos="{SPAWN[0]} {SPAWN[1]} {WHEEL_RADIUS}">
          <freejoint name="base_joint" />
          <inertial pos="0 0 0.055" mass="2.5" diaginertia="0.018 0.018 0.025" />
          <geom name="base_collision" type="cylinder" size="0.092 0.028" pos="0 0 0.030"
                rgba="0.08 0.11 0.13 1" friction="0.9 0.02 0.002" />
          <geom name="lower_plate" type="cylinder" size="0.088 0.010" pos="0 0 0.063"
                rgba="0.05 0.08 0.09 1" contype="0" conaffinity="0" />
          <geom name="upper_plate" type="cylinder" size="0.078 0.010" pos="0 0 0.145"
                rgba="0.08 0.68 0.77 1" contype="0" conaffinity="0" />
          <geom name="mast" type="cylinder" size="0.010 0.052" pos="-0.020 0 0.105"
                rgba="0.65 0.72 0.74 1" contype="0" conaffinity="0" />
          <geom name="lidar" type="cylinder" size="0.042 0.018" pos="0.015 0 0.178"
                rgba="0.05 0.08 0.09 1" contype="0" conaffinity="0" />
          <geom name="camera_case" type="box" size="0.022 0.030 0.018" pos="0.067 0 0.162"
                rgba="0.14 0.18 0.19 1" contype="0" conaffinity="0" />
          <camera name="{CAMERA_NAME}" mode="fixed" pos="0.090 0 0.165"
                  xyaxes="0 -1 0 0 0 1" fovy="79.865" />
          <camera name="{THIRD_PERSON_CAMERA_NAME}" mode="fixed" pos="-0.65 0 0.50"
                  xyaxes="0 -1 0 0.38 0 0.925" fovy="65" />
          <body name="left_wheel" pos="0 0.080 0">
            <joint name="left_wheel_joint" type="hinge" axis="0 1 0" />
            <geom name="left_wheel_geom" type="cylinder" size="{WHEEL_RADIUS} 0.012"
                  euler="{math.pi / 2} 0 0" mass="0.08" rgba="0.035 0.045 0.048 1"
                  friction="2.0 0.01 0.001" solref="0.004 1" />
          </body>
          <body name="right_wheel" pos="0 -0.080 0">
            <joint name="right_wheel_joint" type="hinge" axis="0 1 0" />
            <geom name="right_wheel_geom" type="cylinder" size="{WHEEL_RADIUS} 0.012"
                  euler="{math.pi / 2} 0 0" mass="0.08" rgba="0.035 0.045 0.048 1"
                  friction="2.0 0.01 0.001" solref="0.004 1" />
          </body>
          <body name="caster" pos="-0.073 0 -0.018">
            <geom name="caster_geom" type="sphere" size="0.015" mass="0.025"
                  rgba="0.32 0.36 0.37 1" friction="0.05 0.001 0.0001" />
          </body>
        </body>
        """
    )


def build_model(source: Path | None = None) -> mujoco.MjModel:
    root = load_scene_xml(source)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MolmoSpaces scene has no worldbody")
    worldbody.append(_robot_body())
    return compile_model(root)


class TurtleBotBackend:
    name = "TurtleBot"
    camera_name = CAMERA_NAME

    def __init__(self, source: Path | None = None) -> None:
        self.model = build_model(source)
        self.data = mujoco.MjData(self.model)
        self._left_joint = self.model.joint("left_wheel_joint").id
        self._right_joint = self.model.joint("right_wheel_joint").id
        self._base_joint = self.model.joint("base_joint").id
        self._base_body = self.model.body("base_link").id
        self._base_qpos = self.model.jnt_qposadr[self._base_joint]
        self._base_dof = self.model.jnt_dofadr[self._base_joint]
        self._left_qpos = self.model.jnt_qposadr[self._left_joint]
        self._right_qpos = self.model.jnt_qposadr[self._right_joint]
        self._left_dof = self.model.jnt_dofadr[self._left_joint]
        self._right_dof = self.model.jnt_dofadr[self._right_joint]
        self._robot_geoms = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if self.model.body_rootid[self.model.geom_bodyid[geom_id]]
            == self._base_body
        )
        self._yaw = 0.0
        self._collision_blocked = False
        self._collision_events = 0
        self._collision_contacts: tuple[str, ...] = ()
        self._collision_penetration_m = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._yaw = 0.0
        self._collision_blocked = False
        self._collision_events = 0
        self._collision_contacts = ()
        self._collision_penetration_m = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_planar_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        *,
        data: mujoco.MjData | None = None,
    ) -> None:
        target = self.data if data is None else data
        target.qpos[self._base_qpos] = x
        target.qpos[self._base_qpos + 1] = y
        target.qpos[self._base_qpos + 3 : self._base_qpos + 7] = (
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        )

    def _obstacle_contacts(
        self,
        data: mujoco.MjData | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Return penetrating robot/environment contacts with a lateral normal.

        The kinematic backend must ignore the wheel/floor support contacts, whose
        normal is vertical, while treating walls and obstacle sides as blockers.
        """
        source = self.data if data is None else data
        contacts: list[tuple[str, float]] = []
        for index in range(source.ncon):
            contact = source.contact[index]
            geom0, geom1 = (int(contact.geom[0]), int(contact.geom[1]))
            geom0_is_robot = geom0 in self._robot_geoms
            geom1_is_robot = geom1 in self._robot_geoms
            if geom0_is_robot == geom1_is_robot:
                continue
            normal_xy = math.hypot(float(contact.frame[0]), float(contact.frame[1]))
            penetration_m = max(0.0, -float(contact.dist))
            if normal_xy < OBSTACLE_NORMAL_XY_MIN or penetration_m <= COLLISION_EPS_M:
                continue
            robot_geom = geom0 if geom0_is_robot else geom1
            obstacle_geom = geom1 if geom0_is_robot else geom0
            robot_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                robot_geom,
            ) or f"geom_{robot_geom}"
            obstacle_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                obstacle_geom,
            ) or f"geom_{obstacle_geom}"
            contacts.append((f"{robot_name}:{obstacle_name}", penetration_m))
        return tuple(contacts)

    def plan_world_path(
        self,
        start_pose: Sequence[float],
        goal_world: Sequence[float],
        *,
        resolution_m: float = PLANNER_RESOLUTION_M,
        padding_m: float = PLANNER_PADDING_M,
    ) -> GridPlan:
        """Build a local collision map and plan a world-frame PointGoal path.

        Collision sampling uses a separate ``MjData`` instance. The live robot
        pose, commands and simulation clock are therefore never touched by map
        construction.
        """
        started_s = perf_counter()
        pose = tuple(float(value) for value in start_pose)
        goal = tuple(float(value) for value in goal_world)
        resolution = float(resolution_m)
        padding = float(padding_m)
        if len(pose) != 3 or len(goal) != 2 or not all(
            math.isfinite(value) for value in (*pose, *goal, resolution, padding)
        ):
            raise ValueError("PointGoal planner inputs must be finite")
        if resolution <= 0.0 or padding <= 0.0:
            raise ValueError("PointGoal grid resolution and padding must be positive")

        padding_cells = math.ceil(padding / resolution)
        left_cells = math.ceil((pose[0] - min(pose[0], goal[0])) / resolution)
        lower_cells = math.ceil((pose[1] - min(pose[1], goal[1])) / resolution)
        origin_x = pose[0] - (left_cells + padding_cells) * resolution
        origin_y = pose[1] - (lower_cells + padding_cells) * resolution
        max_x = max(pose[0], goal[0]) + padding
        max_y = max(pose[1], goal[1]) + padding
        width = math.ceil((max_x - origin_x) / resolution) + 1
        height = math.ceil((max_y - origin_y) / resolution) + 1

        probe = mujoco.MjData(self.model)
        blocked: set[tuple[int, int]] = set()
        for row in range(height):
            y = origin_y + row * resolution
            for column in range(width):
                x = origin_x + column * resolution
                for yaw in PLANNER_YAWS:
                    self._set_planar_pose(x, y, yaw, data=probe)
                    mujoco.mj_forward(self.model, probe)
                    if self._obstacle_contacts(probe):
                        blocked.add((row, column))
                        break

        grid = OccupancyGrid(
            origin_x=origin_x,
            origin_y=origin_y,
            resolution_m=resolution,
            width=width,
            height=height,
            blocked=frozenset(blocked),
        )
        start = grid.world_to_cell(pose[:2])
        goal_cell = grid.world_to_cell(goal)
        raw = astar_path(grid, start, goal_cell)
        simplified = simplify_cell_path(grid, raw)
        vertices: list[tuple[float, float]] = [pose[:2]]
        vertices.extend(grid.cell_to_world(cell) for cell in simplified[1:])
        if math.hypot(vertices[-1][0] - goal[0], vertices[-1][1] - goal[1]) > 1e-9:
            vertices.append(goal)
        dense = sample_world_polyline(vertices, spacing_m=resolution)
        if not dense:
            raise ValueError("PointGoal planner produced an empty path")
        first_yaw = dense[0][2]
        world_path = ((pose[0], pose[1], first_yaw), *dense)
        return GridPlan(
            world_path=world_path,
            raw_cells=raw,
            simplified_cells=simplified,
            planning_ms=(perf_counter() - started_s) * 1000.0,
            grid_resolution_m=resolution,
            grid_width=width,
            grid_height=height,
            blocked_cells=len(blocked),
        )

    @staticmethod
    def _max_penetration(contacts: tuple[tuple[str, float], ...]) -> float:
        return max((penetration for _, penetration in contacts), default=0.0)

    @staticmethod
    def _penetration_by_pair(
        contacts: tuple[tuple[str, float], ...],
    ) -> dict[str, float]:
        depths: dict[str, float] = {}
        for name, penetration in contacts:
            depths[name] = max(depths.get(name, 0.0), penetration)
        return depths

    @classmethod
    def _penetration_increased(
        cls,
        candidate: tuple[tuple[str, float], ...],
        baseline: tuple[tuple[str, float], ...],
    ) -> bool:
        baseline_depths = cls._penetration_by_pair(baseline)
        return any(
            penetration > baseline_depths.get(name, 0.0) + COLLISION_EPS_M
            for name, penetration in cls._penetration_by_pair(candidate).items()
        )

    def _set_velocity(self, linear: float, angular: float) -> None:
        left = (linear - angular * WHEEL_TRACK / 2.0) / WHEEL_RADIUS
        right = (linear + angular * WHEEL_TRACK / 2.0) / WHEEL_RADIUS
        self.data.qvel.fill(0.0)
        self.data.qvel[self._base_dof] = linear * math.cos(self._yaw)
        self.data.qvel[self._base_dof + 1] = linear * math.sin(self._yaw)
        self.data.qvel[self._base_dof + 5] = angular
        self.data.qvel[self._left_dof] = left
        self.data.qvel[self._right_dof] = right

    def step(self, command: Twist2D) -> None:
        linear, angular = (float(value) for value in command)
        previous_x = float(self.data.qpos[self._base_qpos])
        previous_y = float(self.data.qpos[self._base_qpos + 1])
        previous_yaw = self._yaw
        baseline_contacts = self._obstacle_contacts()
        candidate_yaw = math.atan2(
            math.sin(previous_yaw + angular * PHYSICS_DT),
            math.cos(previous_yaw + angular * PHYSICS_DT),
        )
        heading = previous_yaw + angular * PHYSICS_DT / 2.0
        candidate_x = previous_x + linear * math.cos(heading) * PHYSICS_DT
        candidate_y = previous_y + linear * math.sin(heading) * PHYSICS_DT

        self._set_planar_pose(candidate_x, candidate_y, candidate_yaw)
        mujoco.mj_forward(self.model, self.data)
        candidate_contacts = self._obstacle_contacts()
        candidate_penetration = self._max_penetration(candidate_contacts)
        blocked = self._penetration_increased(candidate_contacts, baseline_contacts)

        executed_linear = linear
        executed_angular = angular
        if blocked:
            # A forward command may still contain the turn needed to escape the
            # obstacle. Try that rotation at the old position before stopping it.
            self._set_planar_pose(previous_x, previous_y, candidate_yaw)
            mujoco.mj_forward(self.model, self.data)
            rotation_contacts = self._obstacle_contacts()
            if self._penetration_increased(rotation_contacts, baseline_contacts):
                self._set_planar_pose(previous_x, previous_y, previous_yaw)
                executed_angular = 0.0
            executed_linear = 0.0
            self._yaw = (
                candidate_yaw if executed_angular != 0.0 else previous_yaw
            )
            if not self._collision_blocked:
                self._collision_events += 1
            self._collision_blocked = True
            self._collision_contacts = tuple(
                sorted({name for name, _ in candidate_contacts})
            )
            self._collision_penetration_m = candidate_penetration
        else:
            self._yaw = candidate_yaw
            self._collision_blocked = False
            self._collision_contacts = ()
            self._collision_penetration_m = 0.0

        left = (
            executed_linear - executed_angular * WHEEL_TRACK / 2.0
        ) / WHEEL_RADIUS
        right = (
            executed_linear + executed_angular * WHEEL_TRACK / 2.0
        ) / WHEEL_RADIUS
        self.data.qpos[self._left_qpos] += left * PHYSICS_DT
        self.data.qpos[self._right_qpos] += right * PHYSICS_DT
        self._set_velocity(executed_linear, executed_angular)
        self.data.time += PHYSICS_DT
        mujoco.mj_forward(self.model, self.data)

    def state(self) -> RobotState:
        qpos = self.data.qpos
        x, y, z = (
            float(value)
            for value in qpos[self._base_qpos : self._base_qpos + 3]
        )
        qw, qx, qy, qz = (
            float(value)
            for value in qpos[self._base_qpos + 3 : self._base_qpos + 7]
        )
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        left = float(self.data.qvel[self._left_dof])
        right = float(self.data.qvel[self._right_dof])
        linear = WHEEL_RADIUS * (left + right) / 2.0
        angular = WHEEL_RADIUS * (right - left) / WHEEL_TRACK
        return RobotState(
            (x, y, z, yaw),
            (linear, angular),
            telemetry={
                "collision_guard": {
                    "blocked": self._collision_blocked,
                    "events": self._collision_events,
                    "contacts": list(self._collision_contacts),
                    "attempted_penetration_m": self._collision_penetration_m,
                }
            },
        )

    def third_person_camera(self, _camera: mujoco.MjvCamera) -> RenderCamera:
        return THIRD_PERSON_CAMERA_NAME
