"""Isolated, offline language-only PTQ diagnostic; never deploys a candidate.

Uses navigation video replay for calibration and a separate replay for comparison.
Fake-quant HF latency is NOT native kernel performance. No QAT is performed.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


def rvq_tokens(raw):
    """Separate action-code agreement from grounding-token agreement."""
    tokens = re.findall(r"<act_l([0-2])_(\d+)>", raw)
    if [level for level, _ in tokens] != ["0", "1", "2"]:
        raise ValueError("expected exactly three ordered RVQ action tokens")
    return [int(code) for _, code in tokens]


def language_quant_config(stock):
    """Use the recipe's own quantizer attributes, whitelist language layers only."""
    config = copy.deepcopy(stock)
    rules = config["quant_cfg"]
    if not isinstance(rules, list):
        raise ValueError("expected ModelOpt 0.45 ordered quantization rules")
    selected = [{"quantizer_name": "*", "enable": False}]
    for suffix in ("weight_quantizer", "input_quantizer"):
        matches = [r for r in rules if r.get("quantizer_name") == "*" + suffix and "cfg" in r]
        if len(matches) != 1:
            raise ValueError(f"recipe lacks unambiguous {suffix}")
        selected.append({"quantizer_name": "*language_model.layers.*." + suffix,
                         "cfg": copy.deepcopy(matches[0]["cfg"]), "enable": True})
    config["quant_cfg"] = selected
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model-dir", "flash-source", "calibration-dir", "evaluation-dir", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--method", choices=("fp8", "nvfp4"), default="fp8")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 24:
        parser.error("steps must be 1..24")
    for path in (args.model_dir, args.flash_source, args.calibration_dir, args.evaluation_dir):
        if not path.is_absolute() or not path.is_dir():
            parser.error("existing absolute input directories required")
    if args.calibration_dir.resolve() == args.evaluation_dir.resolve():
        parser.error("calibration and evaluation recordings must be separate")
    if not args.output_dir.is_absolute() or args.output_dir.exists():
        parser.error("output-dir must be fresh and absolute")
    sys.path.insert(0, str(args.flash_source))
    from scripts.lightnav_replay_inputs import load_inputs

    calibration, calibration_meta = load_inputs(args.calibration_dir, args.steps)
    evaluation, evaluation_meta = load_inputs(args.evaluation_dir, args.steps)
    os.environ.update(HF_HUB_OFFLINE="1", VLN_EVAL_TRAJ_TOP1="1", VLN_EVAL_TEMPERATURE="0")
    import modelopt
    import modelopt.torch.quantization as mtq
    import numpy as np
    import torch
    import transformers

    from lightnav.tracking import build_tracking_agent

    if modelopt.__version__ != "0.45.0" or transformers.__version__ != "5.8.0":
        raise RuntimeError("requires isolated ModelOpt 0.45.0 and transformers 5.8.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    config = language_quant_config(mtq.FP8_DEFAULT_CFG if args.method == "fp8" else mtq.NVFP4_DEFAULT_CFG)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "lightnav-language-ptq-candidate/v1", "method": args.method,
              "qat": False, "promoted": False, "native_performance_measured": False,
              "calibration": calibration_meta, "evaluation": evaluation_meta,
              "independent_scene_generalization": False,
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                           "modelopt": modelopt.__version__},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        agent = build_tracking_agent(model_path=str(args.model_dir), backend="hf",
                                     device="cuda", task_key="vlnce", max_new_tokens=5)
        agent.engine.bundle.model.to(dtype=torch.float16)

        def replay(frames, metadata, phase):
            instruction = metadata["instruction"]
            agent.reset(instruction=instruction)
            rows = []
            for index, frame in enumerate(frames):
                agent.observe(frame)
                points, raw, latency = agent.predict_waypoints(instruction, task_type="vlnce_traj")
                points = np.asarray(points)
                if points.shape != (10, 3) or not np.isfinite(points).all():
                    raise RuntimeError("invalid candidate trajectory")
                rows.append({"step": index + 1, "raw": raw, "rvq_codes": rvq_tokens(raw),
                             "token_ids": agent.engine.bundle.processor.tokenizer.encode(
                    raw, add_special_tokens=False), "waypoints": points.tolist(), "fake_quant_hf_latency_ms": latency})
                print(json.dumps({"phase": phase, "step": index + 1, "raw": raw}), flush=True)
            return rows

        report["baseline_fp16"] = replay(evaluation, evaluation_meta, "baseline")

        def calibrate(model):
            agent.engine.bundle.model = model
            replay(calibration, calibration_meta, "calibration")

        model = mtq.quantize(agent.engine.bundle.model, config, forward_loop=calibrate)
        agent.engine.bundle.model = model
        enabled = [name for name, module in model.named_modules()
                   if isinstance(module, mtq.nn.TensorQuantizer) and module.is_enabled]
        if not enabled or any("language_model.layers." not in name for name in enabled):
            raise RuntimeError(f"unexpected quantization scope: {enabled}")
        report["enabled_quantizers"] = enabled
        report["candidate"] = replay(evaluation, evaluation_meta, "candidate")
        report["exact_token_matches"] = sum(a["token_ids"] == b["token_ids"] for a, b in zip(
            report["baseline_fp16"], report["candidate"]))
        report["exact_rvq_matches"] = sum(a["rvq_codes"] == b["rvq_codes"] for a, b in zip(
            report["baseline_fp16"], report["candidate"]))
        report["max_xy_error_m"] = max(float(np.linalg.norm(
            np.asarray(a["waypoints"])[:, :2] - np.asarray(b["waypoints"])[:, :2], axis=1).max())
            for a, b in zip(report["baseline_fp16"], report["candidate"]))
        if args.export:
            from modelopt.torch.export import export_hf_checkpoint

            destination = args.output_dir / "checkpoint"
            with torch.inference_mode():
                export_hf_checkpoint(model, export_dir=str(destination))
            agent.engine.bundle.processor.save_pretrained(destination)
            for name in ("eval_config.json", "generation_config.json"):
                source = args.model_dir / name
                if source.is_file() and not (destination / name).exists():
                    shutil.copy2(source, destination / name)
            shutil.copytree(args.model_dir / "action_tokenizer", destination / "action_tokenizer")
            report["exported_checkpoint"] = str(destination)
        report["completed"] = True
    except Exception as error:  # noqa: BLE001 - preserve partial diagnostic evidence
        report["error"] = str(error)
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
