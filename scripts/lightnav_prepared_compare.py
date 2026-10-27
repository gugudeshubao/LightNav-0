"""Offline comparison of recorded Orin tensors; never sends robot commands."""
import argparse
import base64
import json
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from websockets.sync.client import connect

from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor
from lightnav.serving.preprocessed import CONTRACT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--server", default="ws://127.0.0.1:19890")
    args = parser.parse_args()
    if not args.report.is_absolute() or args.report.exists():
        parser.error("fresh absolute report required")
    files = sorted(args.frames.glob("*.png"))
    if not files:
        parser.error("no source frames")
    report = dict(actuation=False, instruction="Walk forward.", runs={})
    try:
        with connect(args.server, max_size=16*1024*1024, open_timeout=5) as ws:
            def request(action, data):
                ws.send(json.dumps(dict(action=action, data=data)))
                reply = json.loads(ws.recv(timeout=60))
                assert reply["action"] == action and reply["data"]["rc"] == 0, reply
                return reply["data"]

            request("login", dict(clientId="prepared-comparison"))
            for mode in ("png", "reference_fp16", "orin_fp16", "png_repeat"):
                request("reset", {})
                rows = report["runs"][mode] = []
                for seq, frame in enumerate(files, 1):
                    data = dict(seq=seq, instruction=report["instruction"])
                    if mode.startswith("png"):
                        data["image"] = base64.b64encode(frame.read_bytes()).decode()
                        action = "next"
                    else:
                        rgb = np.asarray(Image.open(frame).convert("RGB"))
                        tensor = np.load(args.tensors/(frame.stem+".npy"))
                        if mode == "reference_fp16":
                            # Offline numerical oracle only, not the deployment path.
                            tensor = rgb_frame_to_model_tensor(rgb, tensor.shape[:2])
                            tensor = tensor.permute(1, 2, 0).numpy().astype("<f2")
                        packed = zlib.compress(tensor.tobytes(), 1)
                        data["frame_tensor"] = dict(contract=CONTRACT, shape=list(tensor.shape),
                            source_hw=list(rgb.shape[:2]), compression="zlib",
                            data_b64=base64.b64encode(packed).decode())
                        action = "next_tensor"
                    started = time.monotonic()
                    pred = request(action, data)
                    rows.append(dict(frame=frame.name, prediction=pred,
                                     roundtrip_ms=(time.monotonic()-started)*1000))
                print(mode, "completed", len(rows), flush=True)
            request("reset", {})
        summary = report["summary"] = {}
        baseline = report["runs"]["png"]
        for name, rows in report["runs"].items():
            delta = np.stack([np.asarray(a["prediction"]["actions"]["actions"])-
                             np.asarray(b["prediction"]["actions"]["actions"])
                             for a, b in zip(rows, baseline)])
            summary[name] = dict(token_equal=sum(a["prediction"]["raw_text"]==b["prediction"]["raw_text"]
                                               for a, b in zip(rows, baseline)),
                stop_equal=sum(a["prediction"]["stop"]==b["prediction"]["stop"] for a, b in zip(rows, baseline)),
                max_xy_delta_m=float(np.linalg.norm(delta[:, :, :2], axis=2).max()))
        report["completed"] = True
        print(json.dumps(summary, indent=2))
    finally:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
