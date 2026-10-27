"""Kill and restart an owned C++ model client across a real TCP outage; no actuators."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from cloud_fault_relay import FaultRelay
from cloud_realtime_probe import exchange


def run(args):
    report = {"schema": "lightnav-client-restart/v1", "hardware_actuation": False,
              "capture_age_ms": 500, "navigation_success_claimed": False}
    fixture = json.loads(args.input.read_text())
    if fixture["schema"] != "soma-lightnav-simulation-input/v1" or fixture["actuation"] is not False:
        raise ValueError("non-actuating fixture required")
    relay = FaultRelay()
    processes = []
    journal_dir = args.output_dir / "private-state"
    journal_dir.mkdir(mode=0o700)
    journal = journal_dir / "session.json"

    def spawn():
        process = subprocess.Popen(
            [str(args.client), f"http://127.0.0.1:{relay.port}", str(journal)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        processes.append(process)
        return process

    def infer(process):
        stamp = time.monotonic_ns()
        return exchange(process, {"action": "infer", "request_id": 1, "instruction": "Go forward",
                                  "local_capture_stamp_ns": stamp,
                                  "observations": [{"rgb_path": fixture["rgb_path"],
                                                    "width": fixture["width"], "height": fixture["height"],
                                                    "stamp_ns": stamp}]})

    try:
        first = spawn()
        report["first_reset"] = exchange(first, {"action": "reset", "session_id": "restart-first"})
        if not report["first_reset"]["ok"]:
            raise RuntimeError("first reset failed")
        old = json.loads(journal.read_text())
        report["first_infer"] = infer(first)
        if not report["first_infer"]["ok"]:
            raise RuntimeError("first inference failed")
        relay.set_down(True)
        report["offline_close"] = exchange(first, {"action": "close"})
        if report["offline_close"]["ok"]:
            raise RuntimeError("offline cancellation incorrectly confirmed")
        report["persisted_after_failed_cancel"] = json.loads(journal.read_text())["state"]
        first.kill()  # Only the exact subprocess created above; intentionally skips its destructor.
        report["first_exit"] = first.wait(timeout=5)
        relay.set_down(False)
        second = spawn()
        report["second_reset"] = exchange(second, {"action": "reset", "session_id": "restart-second"})
        if not report["second_reset"]["ok"]:
            raise RuntimeError("restart recovery failed")
        new = json.loads(journal.read_text())
        report["new_session_identity"] = new["session"]["session_id"] != old["session"]["session_id"]
        report["second_infer"] = infer(second)
        report["close"] = exchange(second, {"action": "close"})
        second.stdin.close()
        report["second_exit"] = second.wait(timeout=5)
        report["final_journal_state"] = json.loads(journal.read_text())["state"]
        report["execution_passed"] = (
            report["persisted_after_failed_cancel"] == "active"
            and report["first_exit"] == -9 and report["second_exit"] == 0
            and report["new_session_identity"] and report["second_infer"]["ok"]
            and report["close"]["ok"] and report["final_journal_state"] == "closed")
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        report["execution_passed"] = False
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        relay.close()
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report))
    return 0 if report["execution_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(not p.is_absolute() or not p.is_file() for p in (args.client, args.input)):
        parser.error("existing absolute client and fixture required")
    if not args.output_dir.is_absolute() or args.output_dir.exists():
        parser.error("new absolute output-dir required")
    args.output_dir.mkdir(mode=0o700, parents=True)
    os.umask(0o077)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
