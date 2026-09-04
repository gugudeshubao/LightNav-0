from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from pathlib import Path
from urllib.parse import urlparse

from aiohttp import WSMsgType, web

from .avoidance import LocalCollisionRecovery
from .model import SCENE_DATASET, SCENE_ID, SCENE_NAME
from .mpc import (
    A_MAX_V,
    A_MAX_W,
    CONTROL_RATE_HZ,
    HORIZON,
    MPC_DT_S,
    OBJNAV_V_MAX,
    ODOM_MATCH_MAX_GAP_S,
    ODOM_TIMEOUT_S,
    Q_WEIGHTS,
    R_WEIGHTS,
    TRACK_V_MAX,
    W_MAX,
    WAYPOINT_DT_S,
    MpcTracker,
)
from .point_goal import PointGoalState
from .simulation import Simulation
from .vln_client import VlnClient

WEB_DIR = Path(__file__).resolve().parent / "web"


def empty_vln_result() -> dict[str, object]:
    return {
        "latency_ms": None,
        "visible": None,
        "stop": None,
        "apos_state": None,
        "opos_state": None,
        "apos_px": None,
        "opos_px": None,
    }


class VlnMujocoServer:
    def __init__(self, *, default_vln_server: str = "", simulation: Simulation | None = None) -> None:
        self.simulation = simulation or Simulation()
        self.vln = VlnClient(logging.getLogger("vln_mujoco.vln"), default_vln_server)
        self.sockets: set[web.WebSocketResponse] = set()
        self.manual_owner: web.WebSocketResponse | None = None
        self.auto_owner: web.WebSocketResponse | None = None
        self.auto_mode = ""
        self.update_task: asyncio.Task | None = None
        self.last_frame_stamp = 0
        self.waypoints: tuple[tuple[float, float, float], ...] = ()
        self.last_result = empty_vln_result()
        self.auto_command = (0.0, 0.0)
        self.effective_command = (0.0, 0.0)
        self.mpc = MpcTracker()
        self.avoidance = LocalCollisionRecovery()
        self.point_goal = PointGoalState()
        self.point_plan_task: asyncio.Task | None = None
        self._point_goal_generation = 0
        self.capture_poses: dict[int, tuple[float, float, float]] = {}
        self._last_runtime_signature: tuple[object, ...] | None = None

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.websocket)
        app.router.add_get("/api/health", self.health)
        app.router.add_get("/api/camera.jpg", self.camera)
        app.router.add_get("/api/third-person.jpg", self.third_person_camera)
        app.router.add_static("/static", WEB_DIR)
        app.on_startup.append(self.on_startup)
        app.on_shutdown.append(self.on_shutdown)
        return app

    async def on_startup(self, _app: web.Application) -> None:
        self.simulation.start()
        self.update_task = asyncio.create_task(self.update_loop())

    async def on_shutdown(self, _app: web.Application) -> None:
        self._cancel_point_plan()
        if self.update_task is not None:
            self.update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.update_task
        self.vln.stop()
        self.vln.close()
        self.mpc.close()
        self.simulation.set_velocity(0.0, 0.0)
        self.simulation.stop()
        for socket in tuple(self.sockets):
            await socket.close(code=1001, message=b"server shutdown")

    async def index(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB_DIR / "index.html")

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"ok": True, **self.snapshot(None)})

    async def camera(self, _request: web.Request) -> web.Response:
        frame = self.simulation.frame()
        return self.camera_response(frame)

    async def third_person_camera(self, _request: web.Request) -> web.Response:
        frame = self.simulation.third_person_frame()
        return self.camera_response(frame)

    @staticmethod
    def camera_response(frame) -> web.Response:
        if frame is None:
            raise web.HTTPServiceUnavailable(text="camera is warming up")
        return web.Response(
            body=frame.jpeg,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            raise web.HTTPForbidden(text="invalid WebSocket origin")
        socket = web.WebSocketResponse(heartbeat=20.0)
        await socket.prepare(request)
        self.sockets.add(socket)
        await socket.send_json({"type": "snapshot", "data": self.snapshot(socket)})
        try:
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = message.json()
                except Exception:
                    await self.result(socket, False, "invalid JSON")
                    continue
                if isinstance(payload, dict):
                    await self.handle_message(socket, payload)
        finally:
            self.sockets.discard(socket)
            if self.manual_owner is socket:
                self.manual_owner = None
                self.simulation.set_velocity(0.0, 0.0)
            if self.auto_owner is socket:
                self.stop_vln()
            await self.broadcast()
        return socket

    async def handle_message(self, socket: web.WebSocketResponse, payload: dict) -> None:
        kind = str(payload.get("type", ""))
        if kind == "set_server_url":
            try:
                url = self.vln.set_server_url(str(payload.get("server_url", "")))
            except ValueError as exc:
                await self.result(socket, False, str(exc))
            else:
                message = f"VLN Server: {url}" if url else "VLN Server configuration cleared"
                await self.result(socket, True, message, server_url=url)
            return
        if kind == "set_vln":
            if not bool(payload.get("enabled")):
                self.stop_vln()
                await self.result(socket, True, "VLN stopped")
                return
            if self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket}:
                await self.result(socket, False, "another page owns robot control")
                return
            instruction = str(payload.get("instruction", "")).strip()
            if len(instruction) > 500:
                await self.result(socket, False, "instruction is too long")
                return
            try:
                self.vln.start(instruction)
            except ValueError as exc:
                await self.result(socket, False, str(exc))
                return
            self._cancel_point_plan()
            if self.manual_owner is not None:
                self.manual_owner = None
            self.auto_owner = socket
            self.auto_mode = "vln"
            self.auto_command = (0.0, 0.0)
            self.effective_command = (0.0, 0.0)
            self.waypoints = ()
            self.last_result = empty_vln_result()
            self.capture_poses.clear()
            self.mpc.reset()
            self.avoidance.reset()
            self.point_goal.reset()
            self.simulation.set_velocity(0.0, 0.0)
            await self.result(socket, True, "VLN started")
            return
        if kind == "set_point_goal":
            if not bool(payload.get("enabled")):
                if self.auto_mode == "point_goal":
                    self.stop_vln()
                await self.result(socket, True, "PointGoal stopped")
                return
            if self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket}:
                await self.result(socket, False, "another page owns robot control")
                return
            try:
                forward_m = float(payload.get("forward_m"))
                left_m = float(payload.get("left_m"))
                planner = str(payload.get("planner", "direct"))
                simulation = self.simulation.snapshot()
                pose = simulation["pose"]
                start_pose = (pose["x"], pose["y"], pose["yaw"])
                self._cancel_point_plan()
                self.point_goal.start(
                    start_pose,
                    forward_m,
                    left_m,
                    planner=planner,
                )
            except (TypeError, ValueError) as exc:
                await self.result(socket, False, str(exc))
                return
            self.vln.stop()
            if self.manual_owner is not None:
                self.manual_owner = None
            self.auto_owner = socket
            self.auto_mode = "point_goal"
            self.auto_command = (0.0, 0.0)
            self.effective_command = (0.0, 0.0)
            self.waypoints = ()
            self.last_result = empty_vln_result()
            self.capture_poses.clear()
            self.mpc.reset()
            self.avoidance.reset()
            self.simulation.set_velocity(0.0, 0.0)
            goal_world = self.point_goal.goal_world
            if goal_world is None:
                self.point_goal.fail("PointGoal has no world-frame target")
                self.auto_owner = None
                self.auto_mode = ""
                await self.result(socket, False, self.point_goal.error)
                return
            if planner == "direct":
                self.mpc.set_world_path(self.point_goal.world_path)
                await self.result(
                    socket,
                    True,
                    "PointGoal started with direct path",
                    goal_world=self.point_goal.snapshot(start_pose)["goal_world"],
                )
            else:
                self._point_goal_generation += 1
                generation = self._point_goal_generation
                self.point_plan_task = asyncio.create_task(
                    self.finish_point_goal_plan(
                        socket,
                        generation,
                        start_pose,
                        goal_world,
                    )
                )
                await self.result(socket, True, "Building PointGoal collision map…")
            return
        if kind == "acquire_control":
            if self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket}:
                await self.result(socket, False, "another page owns robot control")
                return
            self.stop_vln()
            self.manual_owner = socket
            await self.result(socket, True, "manual control acquired")
            return
        if kind == "release_control":
            if self.manual_owner is socket:
                self.manual_owner = None
                self.simulation.set_velocity(0.0, 0.0)
            await self.result(socket, True, "manual control released")
            return
        if kind == "twist":
            if self.manual_owner is not socket:
                await self.result(socket, False, "acquire manual control first")
                return
            try:
                linear = float(payload.get("linear", 0.0))
                angular = float(payload.get("angular", 0.0))
            except (TypeError, ValueError):
                await self.result(socket, False, "velocity must be numeric")
                return
            if not all(math.isfinite(value) for value in (linear, angular)):
                await self.result(socket, False, "velocity must be finite")
                return
            self.simulation.set_velocity(linear, angular)
            return
        if kind == "stop":
            self.manual_owner = None
            self.stop_vln()
            self.simulation.set_velocity(0.0, 0.0)
            await self.result(socket, True, "robot stopped")
            return
        if kind == "reset":
            self.manual_owner = None
            self.stop_vln()
            self.point_goal.reset()
            self.simulation.reset()
            await self.result(socket, True, "simulation reset")
            return
        await self.result(socket, False, "unknown command")

    def stop_vln(self) -> None:
        self._cancel_point_plan()
        self.auto_owner = None
        self.auto_mode = ""
        self.auto_command = (0.0, 0.0)
        self.effective_command = (0.0, 0.0)
        self.waypoints = ()
        self.last_result = empty_vln_result()
        self.capture_poses.clear()
        self.mpc.reset()
        self.avoidance.reset()
        self.point_goal.stop()
        self.vln.stop()
        self.simulation.set_velocity(0.0, 0.0)

    def _cancel_point_plan(self) -> None:
        self._point_goal_generation += 1
        task = self.point_plan_task
        self.point_plan_task = None
        if task is not None:
            task.cancel()

    async def finish_point_goal_plan(
        self,
        socket: web.WebSocketResponse,
        generation: int,
        start_pose: tuple[float, float, float],
        goal_world: tuple[float, float],
    ) -> None:
        try:
            plan = await asyncio.to_thread(
                self.simulation.plan_world_path,
                start_pose,
                goal_world,
            )
            if (
                generation != self._point_goal_generation
                or self.auto_owner is not socket
                or self.auto_mode != "point_goal"
            ):
                return
            self.point_goal.install_plan(plan)
            self.mpc.set_world_path(plan.world_path)
            await self.result(
                socket,
                True,
                "PointGoal A* path ready",
                planning_ms=plan.planning_ms,
                path_length_m=plan.path_length_m,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if generation != self._point_goal_generation:
                return
            detail = str(exc)
            if detail == "no collision-free path to PointGoal":
                detail = "target is unreachable in the collision map; choose another goal"
            message = f"PointGoal planning failed: {detail}"
            self.point_goal.fail(message)
            self.auto_owner = None
            self.auto_mode = ""
            self.auto_command = (0.0, 0.0)
            self.effective_command = (0.0, 0.0)
            self.mpc.reset()
            self.avoidance.reset(clear_events=False)
            self.simulation.set_velocity(0.0, 0.0)
            await self.result(socket, False, message)
        finally:
            if generation == self._point_goal_generation:
                self.point_plan_task = None

    async def result(
        self,
        socket: web.WebSocketResponse,
        ok: bool,
        message: str,
        **details: object,
    ) -> None:
        await socket.send_json(
            {"type": "command_result", "ok": ok, "message": message, **details}
        )
        await self.broadcast()

    def snapshot(self, socket: web.WebSocketResponse | None) -> dict[str, object]:
        simulation = self.simulation.snapshot()
        pose = simulation["pose"]
        vln = self.vln.snapshot()
        mpc_state = "ERROR" if self.mpc.error else (
            "RUNNING" if self.auto_owner is not None else "IDLE"
        )
        return {
            "scene": {"id": SCENE_ID, "name": SCENE_NAME, "dataset": SCENE_DATASET},
            "robot": {"name": self.simulation.robot_name, "connected": True},
            "simulation": simulation,
            "vln": {
                "state": vln.state,
                "connected": vln.connected,
                "server_url": vln.server_url,
                "instruction": vln.instruction,
                "sequence": vln.sequence,
                "error": vln.error,
                "waypoints": [list(point) for point in self.waypoints],
                **self.last_result,
            },
            "point_goal": self.point_goal.snapshot(
                (pose["x"], pose["y"], pose["yaw"])
            ),
            "mpc": {
                "state": mpc_state,
                "error": self.mpc.error,
                "solve_ms": self.mpc.solve_ms,
                "command": {
                    "linear": self.auto_command[0],
                    "angular": self.auto_command[1],
                },
                "parameters": {
                    "control_rate_hz": CONTROL_RATE_HZ,
                    "horizon": HORIZON,
                    "mpc_dt_s": MPC_DT_S,
                    "waypoint_dt_s": WAYPOINT_DT_S,
                    "track_v_max": TRACK_V_MAX,
                    "objnav_v_max": OBJNAV_V_MAX,
                    "w_max": W_MAX,
                    "a_max_v": A_MAX_V,
                    "a_max_w": A_MAX_W,
                    "q_x": Q_WEIGHTS[0],
                    "q_y": Q_WEIGHTS[1],
                    "q_yaw": Q_WEIGHTS[2],
                    "r_v": R_WEIGHTS[0],
                    "r_w": R_WEIGHTS[1],
                    "odom_match_max_gap_s": ODOM_MATCH_MAX_GAP_S,
                    "odom_timeout_s": ODOM_TIMEOUT_S,
                },
            },
            "avoidance": self.avoidance.snapshot(),
            "control": {
                "manual": self.manual_owner is socket and socket is not None,
                "auto": self.auto_owner is socket and socket is not None,
                "locked": self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket},
                "source": self.auto_mode if self.auto_owner is not None else (
                    "manual" if self.manual_owner is not None else "idle"
                ),
            },
        }

    async def broadcast(self) -> None:
        self._last_runtime_signature = self.runtime_signature()
        for socket in tuple(self.sockets):
            if socket.closed:
                self.sockets.discard(socket)
                continue
            with contextlib.suppress(ConnectionError, RuntimeError):
                await socket.send_json({"type": "runtime", "data": self.snapshot(socket)})

    def runtime_signature(self) -> tuple[object, ...]:
        simulation = self.simulation.snapshot()
        pose = simulation["pose"]
        velocity = simulation["velocity"]
        command = simulation["command"]
        camera = simulation["camera"]
        vln = self.vln.snapshot()
        point_goal = self.point_goal.snapshot((pose["x"], pose["y"], pose["yaw"]))
        return (
            round(float(simulation["sim_time"]), 1),  # keep the page clock ticking while idle
            *(round(float(pose[key]), 6) for key in ("x", "y", "z", "yaw")),
            *(round(float(velocity[key]), 6) for key in ("linear", "angular")),
            *(round(float(command[key]), 6) for key in ("linear", "angular")),
            bool(camera["ready"]),
            vln.state,
            vln.connected,
            vln.server_url,
            vln.instruction,
            vln.sequence,
            vln.error,
            self.waypoints,
            self.last_result["latency_ms"],
            self.last_result["visible"],
            self.last_result["stop"],
            self.last_result["apos_state"],
            self.last_result["opos_state"],
            self.last_result["apos_px"],
            self.last_result["opos_px"],
            self.mpc.error,
            self.mpc.solve_ms,
            self.auto_command,
            self.effective_command,
            self.avoidance.phase,
            self.avoidance.attempts,
            self.avoidance.error,
            self.auto_mode,
            point_goal["state"],
            point_goal["goal_world"],
            point_goal["distance_m"],
            point_goal["planner"],
            point_goal["planning_ms"],
            point_goal["path_length_m"],
            point_goal["error"],
            self.manual_owner is not None,
            self.auto_owner is not None,
        )

    async def update_loop(self) -> None:
        control_period_s = 1.0 / CONTROL_RATE_HZ
        next_control_s = time.monotonic()
        while True:
            frame = self.simulation.frame()
            if frame is not None and frame.stamp_ns != self.last_frame_stamp:
                if self.vln.offer_frame(frame.stamp_ns, frame.jpeg):
                    self.last_frame_stamp = frame.stamp_ns
                    self.capture_poses[frame.stamp_ns] = frame.pose
                    while len(self.capture_poses) > 32:
                        self.capture_poses.pop(next(iter(self.capture_poses)))
            for result in self.vln.take_results():
                if self.auto_mode != "vln":
                    continue
                self.waypoints = result.waypoints
                self.last_result = {
                    "latency_ms": result.latency_ms,
                    "visible": result.visible,
                    "stop": result.stop,
                    "apos_state": result.apos_state,
                    "opos_state": result.opos_state,
                    "apos_px": result.apos_px,
                    "opos_px": result.opos_px,
                }
                if result.stop is True:
                    self.auto_command = (0.0, 0.0)
                    self.effective_command = (0.0, 0.0)
                    self.simulation.set_velocity(0.0, 0.0)
                    self.auto_owner = None
                    self.auto_mode = ""
                    self.capture_poses.clear()
                    self.mpc.reset()
                    self.avoidance.reset()
                    self.vln.stop()
                    break
                capture_pose = self.capture_poses.pop(result.stamp_ns, None)
                if capture_pose is None:
                    self.mpc.reset()
                    self.mpc.error = "missing robot pose for VLN capture timestamp"
                    self.auto_command = (0.0, 0.0)
                    self.effective_command = (0.0, 0.0)
                    self.avoidance.reset(clear_events=False)
                    self.simulation.set_velocity(0.0, 0.0)
                    continue
                self.mpc.set_body_path(result.waypoints, capture_pose)
                self.auto_command = self.mpc.command

            solved_command = self.mpc.poll()
            if solved_command is not None:
                self.auto_command = solved_command

            now_s = time.monotonic()
            if self.auto_owner is not None and now_s >= next_control_s:
                simulation = self.simulation.snapshot()
                pose = simulation["pose"]
                current_pose = (pose["x"], pose["y"], pose["yaw"])
                reached = self.auto_mode == "point_goal" and self.point_goal.update(
                    current_pose
                )
                if reached:
                    self.auto_command = (0.0, 0.0)
                    self.effective_command = (0.0, 0.0)
                    self.simulation.set_velocity(0.0, 0.0)
                    self.auto_owner = None
                    self.auto_mode = ""
                    self.mpc.reset()
                    self.avoidance.reset(clear_events=False)
                else:
                    self.mpc.submit(current_pose)
                    collision_guard = simulation.get("backend", {}).get(
                        "collision_guard",
                        {},
                    )
                    if self.mpc.error:
                        self.avoidance.reset(clear_events=False)
                        self.effective_command = (0.0, 0.0)
                    else:
                        self.effective_command = self.avoidance.update(
                            self.auto_command,
                            current_pose,
                            collision_blocked=bool(collision_guard.get("blocked")),
                        )
                    self.simulation.set_velocity(*self.effective_command)
                    point_goal_error = self.mpc.error or self.avoidance.error
                    if self.auto_mode == "point_goal" and point_goal_error:
                        self.point_goal.fail(point_goal_error)
                        self.auto_owner = None
                        self.auto_mode = ""
                        self.auto_command = (0.0, 0.0)
                        self.effective_command = (0.0, 0.0)
                        self.simulation.set_velocity(0.0, 0.0)
                        self.mpc.reset()
                while next_control_s <= now_s:
                    next_control_s += control_period_s
            elif self.auto_owner is None:
                next_control_s = now_s
            if self.runtime_signature() != self._last_runtime_signature:
                await self.broadcast()
            await asyncio.sleep(0.01)
