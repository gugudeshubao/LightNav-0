(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const storedBoolean = (key) => {
    try { return localStorage.getItem(key) === "true"; } catch { return false; }
  };
  const storeBoolean = (key, value) => {
    try { localStorage.setItem(key, String(value)); } catch { return; }
  };
  const ui = {
    cameraImage: $("camera-image"), path: $("path-overlay"), thirdPersonOverlay: $("third-person-overlay"),
    thirdPersonOverlayImage: $("third-person-overlay-image"), thirdPersonOverlayToggle: $("third-person-overlay-toggle"),
    firstPersonView: $("first-person-view"), thirdPersonView: $("third-person-view"),
    simTime: $("sim-time"), cameraFps: $("camera-fps"), latency: $("latency"), count: $("waypoint-count"),
    serverUrl: $("server-url"), saveServer: $("save-server"), instruction: $("instruction"), vlnToggle: $("vln-toggle"),
    vlnState: $("vln-state"), vlnDetail: $("vln-detail"), controlToggle: $("control-toggle"), stop: $("stop-button"), reset: $("reset-button"),
    instructionGoalMode: $("instruction-goal-mode"), pointGoalMode: $("point-goal-mode"),
    instructionGoalPanel: $("instruction-goal-panel"), pointGoalPanel: $("point-goal-panel"),
    pointForward: $("point-forward"), pointLeft: $("point-left"), pointPlanner: $("point-planner"), pointGoalToggle: $("point-goal-toggle"),
    pointGoalState: $("point-goal-state"), pointGoalDetail: $("point-goal-detail"),
    toast: $("toast")
  };
  const state = {
    socket:null, connected:false, reconnect:null, data:null, keys:new Set(), toastTimer:null,
    cameraBusy:false, cameraUrl:null, overlayCameraUrl:null, cameraView:"first",
    overlayMinimized:storedBoolean("third-person-overlay-minimized"), serverUrlDirty:false,
    goalMode:"instruction"
  };

  function send(payload) {
    if (state.socket?.readyState !== WebSocket.OPEN) { toast("Web connection not ready", true); return false; }
    state.socket.send(JSON.stringify(payload));
    return true;
  }
  function connect() {
    clearTimeout(state.reconnect);
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/ws`);
    state.socket = socket;
    socket.onopen = () => { state.connected = true; render(); };
    socket.onclose = () => {
      state.connected = false; state.keys.clear(); render();
      state.reconnect = setTimeout(connect, 1000);
    };
    socket.onerror = () => socket.close();
    socket.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      if (["snapshot", "runtime"].includes(message.type)) {
        state.data = message.data;
        if (message.type === "snapshot" && !state.serverUrlDirty && typeof message.data?.vln?.server_url === "string") {
          ui.serverUrl.value = message.data.vln.server_url;
        }
      }
      if (message.type === "command_result") {
        if (message.ok && typeof message.server_url === "string") {
          ui.serverUrl.value = message.server_url;
          state.serverUrlDirty = false;
        }
        toast(message.message || (message.ok ? "Done" : "Failed"), !message.ok);
      }
      render();
    };
  }
  async function refreshImage(image, endpoint, urlKey, isCurrent) {
    let nextUrl = null;
    try {
      const response = await fetch(`${endpoint}?t=${Date.now()}`, {cache:"no-store"});
      if (!response.ok) throw new Error("camera unavailable");
      nextUrl = URL.createObjectURL(await response.blob());
      if (!isCurrent()) return;
      await new Promise((resolve, reject) => {
        image.onload = resolve; image.onerror = reject; image.src = nextUrl;
      });
      if (!isCurrent()) return;
      const oldUrl = state[urlKey]; state[urlKey] = nextUrl; nextUrl = null;
      if (oldUrl) URL.revokeObjectURL(oldUrl);
    } finally {
      if (nextUrl) URL.revokeObjectURL(nextUrl);
    }
  }
  async function refreshCamera() {
    if (state.cameraBusy) return;
    state.cameraBusy = true;
    const cameraView = state.cameraView;
    try {
      const endpoint = cameraView === "third" ? "/api/third-person.jpg" : "/api/camera.jpg";
      const updates = [refreshImage(ui.cameraImage, endpoint, "cameraUrl", () => cameraView === state.cameraView)];
      if (cameraView === "first") {
        updates.push(refreshImage(
          ui.thirdPersonOverlayImage,
          "/api/third-person.jpg",
          "overlayCameraUrl",
          () => state.cameraView === "first"
        ));
      }
      await Promise.allSettled(updates);
    } finally { state.cameraBusy = false; }
  }
  function render() {
    const data = state.data || {};
    const sim = data.simulation || {}, camera = sim.camera || {}, backend = sim.backend || {}, collision = backend.collision_guard || {}, avoidance = data.avoidance || {}, vln = data.vln || {}, pointGoal = data.point_goal || {}, mpc = data.mpc || {}, control = data.control || {};
    const activeVln = control.auto === true && control.source === "vln";
    const activePointGoal = control.auto === true && control.source === "point_goal";
    if (activeVln) state.goalMode = "instruction";
    if (activePointGoal) state.goalMode = "point";
    const showingPointGoal = state.goalMode === "point";
    ui.instructionGoalMode.classList.toggle("active", !showingPointGoal);
    ui.pointGoalMode.classList.toggle("active", showingPointGoal);
    ui.instructionGoalMode.setAttribute("aria-selected", String(!showingPointGoal));
    ui.pointGoalMode.setAttribute("aria-selected", String(showingPointGoal));
    ui.instructionGoalPanel.hidden = showingPointGoal;
    ui.pointGoalPanel.hidden = !showingPointGoal;
    ui.simTime.textContent = `${Number(sim.sim_time || 0).toFixed(1)} s`;
    ui.cameraFps.textContent = camera.ready ? `${Number(camera.fps || 0).toFixed(1)} Hz` : "—";
    ui.firstPersonView.classList.toggle("active", state.cameraView === "first");
    ui.thirdPersonView.classList.toggle("active", state.cameraView === "third");
    ui.thirdPersonOverlay.hidden = state.cameraView !== "first";
    ui.thirdPersonOverlay.classList.toggle("minimized", state.overlayMinimized);
    ui.thirdPersonOverlayToggle.textContent = state.overlayMinimized ? "+" : "−";
    ui.thirdPersonOverlayToggle.setAttribute("aria-label", state.overlayMinimized ? "Restore third-person view" : "Minimize third-person view");
    ui.thirdPersonOverlayToggle.setAttribute("aria-expanded", String(!state.overlayMinimized));
    ui.thirdPersonOverlayToggle.title = state.overlayMinimized ? "Restore third-person view" : "Minimize third-person view";
    const latency = showingPointGoal ? mpc.solve_ms : vln.latency_ms;
    ui.latency.textContent = Number.isFinite(Number(latency)) ? `${Number(latency).toFixed(0)} ms` : "—";
    const waypoints = showingPointGoal
      ? (Array.isArray(pointGoal.waypoints) ? pointGoal.waypoints : [])
      : (Array.isArray(vln.waypoints) ? vln.waypoints : []);
    ui.count.textContent = String(waypoints.length);
    ui.vlnState.textContent = vln.state || "IDLE";
    ui.vlnToggle.textContent = activeVln ? "Stop VLN" : "Start VLN";
    ui.vlnToggle.classList.toggle("active", activeVln);
    ui.vlnToggle.disabled = !state.connected || Boolean(control.locked) || (Boolean(control.auto) && !activeVln);
    ui.vlnDetail.textContent = mpc.error || vln.error || avoidance.error || (avoidance.active === true
      ? `Local collision recovery · ${String(avoidance.phase || "active").toUpperCase()}`
      : (collision.blocked === true
      ? "Motion blocked by collision guard; waiting for a safe path."
      : (vln.stop === true
      ? "VLN reached the goal and stopped automatically."
      : (activeVln
        ? (vln.connected ? (vln.visible === false ? "Target is currently not visible." : `MPC tracking the VLN path${Number.isFinite(Number(mpc.solve_ms)) ? ` · ${Number(mpc.solve_ms).toFixed(1)} ms` : ""}`) : "Connecting to the VLN server…")
        : "Set the server URL to get started."))));
    ui.pointGoalState.textContent = pointGoal.state || "IDLE";
    ui.pointGoalToggle.textContent = activePointGoal ? "Stop PointGoal" : "Start PointGoal";
    ui.pointGoalToggle.classList.toggle("active", activePointGoal);
    ui.pointGoalToggle.disabled = !state.connected || Boolean(control.locked) || (Boolean(control.auto) && !activePointGoal);
    const pointDistance = Number(pointGoal.distance_m);
    let pointDetail = "Enter a target relative to the current robot pose.";
    if (pointGoal.error || avoidance.error) pointDetail = pointGoal.error || avoidance.error;
    else if (avoidance.active === true) pointDetail = `Local collision recovery · ${String(avoidance.phase || "active").toUpperCase()}`;
    else if (collision.blocked === true) pointDetail = "Motion blocked by collision guard; waiting for a safe path.";
    else if (pointGoal.state === "PLANNING") pointDetail = "Sampling collision map and planning A* route…";
    else if (pointGoal.state === "REACHED") pointDetail = `Goal reached within ${Number(pointGoal.tolerance_m || 0).toFixed(2)} m.`;
    else if (activePointGoal && Number.isFinite(pointDistance)) {
      const planner = pointGoal.planner === "astar" ? "A*" : "direct";
      const planMs = Number(pointGoal.planning_ms), pathLength = Number(pointGoal.path_length_m);
      pointDetail = `${planner} · distance ${pointDistance.toFixed(2)} m${Number.isFinite(pathLength) ? ` · path ${pathLength.toFixed(2)} m` : ""}${Number.isFinite(planMs) ? ` · plan ${planMs.toFixed(0)} ms` : ""}${Number.isFinite(Number(mpc.solve_ms)) ? ` · MPC ${Number(mpc.solve_ms).toFixed(1)} ms` : ""}`;
    }
    else if (pointGoal.state === "STOPPED") pointDetail = "PointGoal stopped before arrival.";
    ui.pointGoalDetail.textContent = pointDetail;
    ui.controlToggle.textContent = control.manual ? "Release" : (control.locked ? "In use" : "Take control");
    ui.controlToggle.classList.toggle("active", Boolean(control.manual));
    ui.controlToggle.disabled = !state.connected || (control.locked && !control.manual);
    document.querySelectorAll(".drive-key").forEach((button) => {
      button.disabled = !control.manual;
      button.classList.toggle("active", state.keys.has(button.dataset.key));
    });
    drawPath(waypoints, activeVln || activePointGoal, activeVln ? vln : {}, state.cameraView === "first");
  }
  function drawPath(points, enabled, vln, firstPerson) {
    const context = ui.path.getContext("2d");
    context.clearRect(0, 0, ui.path.width, ui.path.height);
    if (!firstPerson) return;
    if (enabled && points.length) {
      const project = (point) => [240 - Number(point[1]) * 75, 258 - Number(point[0]) * 75];
      context.lineWidth = 3; context.strokeStyle = "#60a5fa"; context.fillStyle = "#60a5fa";
      context.beginPath(); context.moveTo(240, 258);
      points.forEach((point) => { const [x,y] = project(point); context.lineTo(x,y); });
      context.stroke();
      points.forEach((point, index) => { const [x,y] = project(point); context.beginPath(); context.arc(x,y,index === points.length-1 ? 5 : 3,0,Math.PI*2); context.fill(); });
    }
    const drawMarker = (pixel, channel, semanticState, color) => {
      if (!Array.isArray(pixel) || pixel.length !== 2) return;
      const x = Number(pixel[0]), y = Number(pixel[1]);
      if (![x, y].every(Number.isFinite) || x < 0 || x >= 480 || y < 0 || y >= 270) return;
      const label = semanticState ? `${channel} · ${semanticState}` : channel;
      context.save();
      context.strokeStyle = color; context.fillStyle = color; context.lineWidth = 2;
      context.beginPath(); context.arc(x, y, 6, 0, Math.PI * 2); context.stroke();
      context.beginPath(); context.moveTo(x - 9, y); context.lineTo(x + 9, y); context.moveTo(x, y - 9); context.lineTo(x, y + 9); context.stroke();
      context.font = "700 9px ui-monospace, monospace";
      const textWidth = context.measureText(label).width;
      const labelX = Math.max(3, Math.min(480 - textWidth - 9, x + 9));
      const labelY = y < 18 ? y + 18 : y - 8;
      context.fillStyle = "rgba(4, 6, 10, .78)"; context.fillRect(labelX - 3, labelY - 10, textWidth + 6, 13);
      context.fillStyle = color; context.fillText(label, labelX, labelY);
      context.restore();
    };
    drawMarker(vln.apos_px, "APOS", vln.apos_state, "#ffb454");
    drawMarker(vln.opos_px, "OPOS", vln.opos_state, "#6ee7b7");
  }
  function toast(message, error=false) {
    clearTimeout(state.toastTimer); ui.toast.textContent = message; ui.toast.classList.toggle("error", error); ui.toast.classList.add("visible");
    state.toastTimer = setTimeout(() => ui.toast.classList.remove("visible"), 2200);
  }
  const MANUAL_LINEAR = 1.0;   // m/s
  const MANUAL_ANGULAR = 1.0;  // rad/s
  function manualVelocity() {
    return {
      linear: ((state.keys.has("w") ? 1 : 0) - (state.keys.has("s") ? 1 : 0)) * MANUAL_LINEAR,
      angular: ((state.keys.has("a") ? 1 : 0) - (state.keys.has("d") ? 1 : 0)) * MANUAL_ANGULAR
    };
  }
  function publishManual() {
    if (!state.data?.control?.manual) return;
    send({type:"twist", ...manualVelocity()});
  }
  function setKey(key, active) {
    if (!"wasd".includes(key)) return;
    if (active) state.keys.add(key); else state.keys.delete(key);
    publishManual(); render();
  }

  ui.serverUrl.addEventListener("input", () => { state.serverUrlDirty = true; });
  ui.saveServer.addEventListener("click", () => send({type:"set_server_url", server_url:ui.serverUrl.value.trim()}));
  ui.instructionGoalMode.addEventListener("click", () => { state.goalMode = "instruction"; render(); });
  ui.pointGoalMode.addEventListener("click", () => { state.goalMode = "point"; render(); });
  ui.firstPersonView.addEventListener("click", () => { state.cameraView = "first"; render(); refreshCamera(); });
  ui.thirdPersonView.addEventListener("click", () => { state.cameraView = "third"; render(); refreshCamera(); });
  ui.thirdPersonOverlayToggle.addEventListener("click", () => {
    state.overlayMinimized = !state.overlayMinimized;
    storeBoolean("third-person-overlay-minimized", state.overlayMinimized);
    render();
  });
  ui.vlnToggle.addEventListener("click", () => {
    const enabled = !(state.data?.control?.auto === true && state.data?.control?.source === "vln");
    const instruction = ui.instruction.value.trim();
    if (enabled && !instruction) return toast("Enter a navigation instruction", true);
    send({type:"set_vln", enabled, instruction});
  });
  ui.pointGoalToggle.addEventListener("click", () => {
    const enabled = !(state.data?.control?.auto === true && state.data?.control?.source === "point_goal");
    const forward_m = Number(ui.pointForward.value), left_m = Number(ui.pointLeft.value);
    if (enabled && ![forward_m, left_m].every(Number.isFinite)) return toast("Enter finite PointGoal coordinates", true);
    send({type:"set_point_goal", enabled, forward_m, left_m, planner:ui.pointPlanner.value});
  });
  ui.controlToggle.addEventListener("click", () => send({type:state.data?.control?.manual ? "release_control" : "acquire_control"}));
  ui.stop.addEventListener("click", () => { state.keys.clear(); send({type:"stop"}); });
  ui.reset.addEventListener("click", () => { state.keys.clear(); send({type:"reset"}); });
  document.querySelectorAll(".drive-key").forEach((button) => {
    const key = button.dataset.key;
    button.addEventListener("pointerdown", (event) => { event.preventDefault(); button.setPointerCapture(event.pointerId); setKey(key, true); });
    button.addEventListener("pointerup", () => setKey(key, false));
    button.addEventListener("pointercancel", () => setKey(key, false));
  });
  window.addEventListener("keydown", (event) => {
    if (event.repeat || ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if ("wasd".includes(event.key.toLowerCase())) { event.preventDefault(); setKey(event.key.toLowerCase(), true); }
    if (event.key === " ") { event.preventDefault(); state.keys.clear(); send({type:"stop"}); }
  });
  window.addEventListener("keyup", (event) => setKey(event.key.toLowerCase(), false));
  window.addEventListener("blur", () => { state.keys.clear(); publishManual(); render(); });
  setInterval(publishManual, 100);
  setInterval(refreshCamera, 90);
  connect(); render();
})();
