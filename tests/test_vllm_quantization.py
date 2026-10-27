"""vLLM quantization is explicit, optional, and isolated from the HF backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from lightnav.inference import vllm_utils
from lightnav.inference.config import InferenceConfig
from lightnav.inference.engine import build_engine


@pytest.mark.parametrize("method", [None, "fp8", "bitsandbytes"])
def test_load_vllm_engine_forwards_quantization(monkeypatch, method):
    seen = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    monkeypatch.setattr(vllm_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        vllm_utils,
        "resolve_model_paths",
        lambda _path: ("/checkpoint", "/checkpoint", "/checkpoint"),
    )

    config = InferenceConfig(
        model_path="/checkpoint",
        backend="vllm_local",
        vllm_quantization=method,
    )
    vllm_utils.load_vllm_engine(config, num_frames=8)

    assert seen["model"] == "/checkpoint"
    assert seen["dtype"] == "bfloat16"
    assert seen["quantization"] == method
    assert seen["load_format"] == ("bitsandbytes" if method == "bitsandbytes" else "auto")


def test_hf_backend_rejects_vllm_quantization():
    config = InferenceConfig(
        model_path="/checkpoint",
        backend="hf",
        vllm_quantization="fp8",
    )
    with pytest.raises(ValueError, match="requires backend='vllm_local'"):
        build_engine(config)
