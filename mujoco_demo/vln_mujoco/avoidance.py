"""Bounded reactive collision recovery for the MuJoCo VLN demo.

The VLA and MPC still own the navigation intent. This layer only converts a
collision-guard stop into a short, deterministic detour: turn along the obstacle,
advance, turn back toward the original heading, then probe forward. If the probe
is still blocked it repeats one bypass segment, up to a fixed attempt limit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .robots.base import Twist2D

TURN_SPEED_RAD_S = 1.0
FORWARD_SPEED_M_S = 0.3
BACKOFF_SPEED_M_S = 0.2
TURN_ANGLE_RAD = math.pi / 2.0
BACKOFF_DISTANCE_M = 0.12
BYPASS_DISTANCE_M = 0.45
PROBE_DISTANCE_M = 0.30
YAW_TOLERANCE_RAD = 0.06
MAX_ATTEMPTS = 8
MAX_PHASE_TICKS = 200
MIN_FORWARD_COMMAND_M_S = 0.02


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


class LocalCollisionRecovery:
    """Small Bug-style detour state machine with bounded failure behaviour."""

    def __init__(self) -> None:
        self.events = 0
        self.reset(clear_events=False)

    def reset(self, *, clear_events: bool = True) -> None:
        if clear_events:
            self.events = 0
        self.phase = "idle"
        self.command: Twist2D = (0.0, 0.0)
        self.direction = 1
        self.attempts = 0
        self._impact_yaw = 0.0
        self._target_yaw = 0.0
        self._phase_origin = (0.0, 0.0)
        self._phase_ticks = 0
        self.error = ""

    @property
    def active(self) -> bool:
        return self.phase not in {"idle", "failed"}

    def snapshot(self) -> dict[str, object]:
        return {
            "active": self.active,
            "phase": self.phase,
            "command": {
                "linear": self.command[0],
                "angular": self.command[1],
            },
            "direction": "left" if self.direction > 0 else "right",
            "events": self.events,
            "attempts": self.attempts,
            "error": self.error,
        }

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self._phase_ticks = 0

    def _fail(self, message: str) -> Twist2D:
        self.phase = "failed"
        self.command = (0.0, 0.0)
        self.error = message
        return self.command

    def _start_detour(
        self,
        pose: tuple[float, float, float],
        desired: Twist2D,
    ) -> None:
        self.events += 1
        self.attempts = 1
        self.error = ""
        self._impact_yaw = pose[2]
        if abs(desired[1]) > 0.05:
            self.direction = 1 if desired[1] > 0.0 else -1
        else:
            # Deterministic left-hand rule for a symmetric head-on obstacle.
            self.direction = 1
        self._target_yaw = wrap_angle(
            self._impact_yaw + self.direction * TURN_ANGLE_RAD
        )
        self._phase_origin = pose[:2]
        self._set_phase("backoff")

    def _retry_bypass(self, pose: tuple[float, float, float]) -> bool:
        self.attempts += 1
        if self.attempts > MAX_ATTEMPTS:
            self._fail("local collision recovery exhausted its attempt limit")
            return False
        self._target_yaw = wrap_angle(
            self._impact_yaw + self.direction * TURN_ANGLE_RAD
        )
        self._phase_origin = pose[:2]
        self._set_phase("backoff")
        return True

    def _turn_command(self, yaw: float) -> Twist2D | None:
        error = wrap_angle(self._target_yaw - yaw)
        if abs(error) <= YAW_TOLERANCE_RAD:
            return None
        angular = max(
            -TURN_SPEED_RAD_S,
            min(TURN_SPEED_RAD_S, 2.5 * error),
        )
        return 0.0, angular

    def update(
        self,
        desired: Twist2D,
        pose: Sequence[float],
        *,
        collision_blocked: bool,
    ) -> Twist2D:
        desired = (float(desired[0]), float(desired[1]))
        values = tuple(float(value) for value in pose)
        if len(values) != 3 or not all(math.isfinite(value) for value in (*desired, *values)):
            return self._fail("local collision recovery received non-finite state")
        current = (values[0], values[1], wrap_angle(values[2]))

        if self.phase == "failed":
            return self.command
        if self.phase == "idle":
            if not collision_blocked or desired[0] <= MIN_FORWARD_COMMAND_M_S:
                self.command = desired
                return self.command
            self._start_detour(current, desired)

        self._phase_ticks += 1
        if self._phase_ticks > MAX_PHASE_TICKS:
            return self._fail(f"local collision recovery stalled in {self.phase}")

        # A transition can immediately satisfy the next phase too. Bound the
        # loop so malformed state cannot spin here indefinitely.
        for _ in range(4):
            if self.phase == "backoff":
                distance = math.hypot(
                    current[0] - self._phase_origin[0],
                    current[1] - self._phase_origin[1],
                )
                if distance < BACKOFF_DISTANCE_M:
                    self.command = (-BACKOFF_SPEED_M_S, 0.0)
                    return self.command
                self._set_phase("turn_away")
                continue

            if self.phase == "turn_away":
                command = self._turn_command(current[2])
                if command is not None:
                    self.command = command
                    return command
                self._phase_origin = current[:2]
                self._set_phase("bypass")
                continue

            if self.phase == "bypass":
                if collision_blocked:
                    # The selected side is itself obstructed. Reverse the
                    # wall-following direction and try the other side.
                    self.direction *= -1
                    if not self._retry_bypass(current):
                        return self.command
                    continue
                distance = math.hypot(
                    current[0] - self._phase_origin[0],
                    current[1] - self._phase_origin[1],
                )
                if distance < BYPASS_DISTANCE_M:
                    self.command = (FORWARD_SPEED_M_S, 0.0)
                    return self.command
                self._target_yaw = self._impact_yaw
                self._set_phase("turn_forward")
                continue

            if self.phase == "turn_forward":
                command = self._turn_command(current[2])
                if command is not None:
                    self.command = command
                    return command
                self._phase_origin = current[:2]
                self._set_phase("probe")
                continue

            if self.phase == "probe":
                if collision_blocked:
                    if not self._retry_bypass(current):
                        return self.command
                    continue
                distance = math.hypot(
                    current[0] - self._phase_origin[0],
                    current[1] - self._phase_origin[1],
                )
                if distance < PROBE_DISTANCE_M:
                    self.command = (FORWARD_SPEED_M_S, 0.0)
                    return self.command
                self._set_phase("idle")
                self.command = desired
                return self.command

            return self._fail(f"unknown local collision recovery phase: {self.phase}")

        return self._fail("local collision recovery exceeded transition limit")
