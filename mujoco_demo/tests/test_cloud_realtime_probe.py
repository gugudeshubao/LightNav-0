import importlib.util
import sys
from pathlib import Path

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("cloud_probe", scripts / "cloud_realtime_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixture():
    job = {"epoch": 1, "request_id": 2, "stamp": 10}
    reply = {"ok": True, "request_id": 2, "source_stamp_ns": 10,
             "produced_at_ns": 20, "deadline_ns": 30}
    return job, reply


def test_fresh():
    job, reply = fixture()
    assert probe.accept(job, reply, 1, 25)


def test_zero_command_reason_and_endpoint_projection():
    planned = {"command": [0, 0, 0], "mode": "reference",
               "local_path_arrived": True, "obstacle_blocked": False,
               "reference": [[1, 0, 0]]}
    result = probe.planning_diagnostics(planned, (2, 0), False)
    assert result["reason"] == "local_reference_complete"
    assert result["endpoint_distance_m"] == 1
    assert result["endpoint_forward_projection_m"] == -1
    assert probe.planning_diagnostics(planned, (2, 0), True)["reason"] == "expired_after_planning"
    planned["local_path_arrived"] = False
    planned["obstacle_blocked"] = True
    assert probe.planning_diagnostics(planned, (0, 0), False)["reason"] == "planner_blocked"
    planned["mode"] = "model_stop"
    assert probe.planning_diagnostics(planned, (0, 0), False)["reason"] == "model_stop"
    assert probe.planning_diagnostics(None, (0, 0), False)["reason"] == "no_valid_path"


def test_comparison_budget_uses_original_capture_time():
    import pytest
    for budget in (500, 650, 800):
        reply = {"ok": True, "capture_age_limit_ms": budget,
                 "source_stamp_ns": 123, "deadline_ns": 123 + budget * 1_000_000}
        probe.check_capture_budget(reply, budget)
        with pytest.raises(RuntimeError, match="mismatch"):
            probe.check_capture_budget(reply, budget + 1)
        reply["deadline_ns"] += 1
        with pytest.raises(RuntimeError, match="moved"):
            probe.check_capture_budget(reply, budget)


def test_timing_breakdown_and_clock_order():
    import pytest
    job = {"stamp": 1000000, "submitted_ns": 2000000}
    reply = {"probe_exchange_started_ns": 3000000, "probe_reply_received_ns": 8000000}
    assert probe.timing_breakdown(job, reply, 10000000) == {
        "capture_to_submit_ms": 1, "executor_wait_ms": 1,
        "stdio_roundtrip_ms": 5, "reply_to_consume_ms": 2,
        "capture_to_consume_ms": 9}
    with pytest.raises(RuntimeError, match="non-monotonic"):
        probe.timing_breakdown(job, reply, 7000000)


def test_expired_and_future_clock_rejected():
    job, reply = fixture()
    assert probe.accept(job, reply, 1, 30) is None
    assert probe.accept(job, reply, 1, 19) is None


def test_task_switch_discards_old_result():
    job, reply = fixture()
    assert probe.accept(job, reply, 2, 25) is None


def test_failed_reply_not_consumed():
    job, _ = fixture()
    assert probe.accept(job, {"ok": False}, 1, 25) is None


def test_identity_mismatch_fails_closed():
    import pytest
    job, reply = fixture()
    reply["source_stamp_ns"] = 11
    with pytest.raises(RuntimeError, match="identity"):
        probe.accept(job, reply, 1, 25)


def test_reset_retry_is_explicit_and_bounded():
    assert not probe.retry_reset(1, 0)
    assert probe.retry_reset(1, 2)
    assert probe.retry_reset(2, 2)
    assert not probe.retry_reset(3, 2)
    assert not probe.retry_reset(0, 2)


def test_real_tcp_fault_and_restore():
    import socket
    import socketserver
    import threading

    from cloud_fault_relay import FaultRelay

    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            data = self.request.recv(32)
            if data:
                self.request.sendall(data)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Echo)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    relay = FaultRelay(server.server_address[1])
    try:
        for down in (False, True, False):
            relay.set_down(down)
            with socket.create_connection(("127.0.0.1", relay.port), timeout=1) as client:
                try:
                    client.sendall(b"test")
                    reply = client.recv(32)
                except (ConnectionResetError, BrokenPipeError):
                    reply = b""
                assert reply == (b"" if down else b"test")
    finally:
        relay.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
