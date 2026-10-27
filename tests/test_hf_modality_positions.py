"""HF generation must retain multimodal RoPE after sample prompt trimming."""

from types import SimpleNamespace

import pytest
import torch

from lightnav.inference.engine import VLNInferenceEngine


@pytest.mark.parametrize("grid_key", ["video_grid_thw", "image_grid_thw"])
def test_hf_rebuilds_modality_types_from_final_prompt(monkeypatch, grid_key):
    captured = {}

    class FakeModel:
        config = SimpleNamespace(image_token_id=41, video_token_id=42)

        def _validate_model_kwargs(self, kwargs):
            pass

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.cat((kwargs["input_ids"], torch.tensor([[99]])), dim=1)

    engine = VLNInferenceEngine.__new__(VLNInferenceEngine)
    engine.bundle = SimpleNamespace(
        device="cpu",
        model=FakeModel(),
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=99, decode=lambda *a, **k: "ok"),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setenv("VLN_EVAL_TRAJ_TOP1", "0")
    monkeypatch.setenv("VLN_EVAL_TEMPERATURE", "0")
    ids = torch.tensor([7, 41, 42, 7, 99])
    inputs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": torch.tensor([-100, -100, -100, -100, 99]),
        "position_ids": torch.zeros(3, 5, dtype=torch.long),
        # Deliberately stale processor metadata must not reach generate().
        "mm_token_type_ids": torch.zeros(1, 5, dtype=torch.long),
        grid_key: torch.tensor([[1, 2, 2]]),
    }
    text, _ = engine._generate_hf(inputs, max_new_tokens=1)
    assert text == "ok"
    assert "position_ids" not in captured
    assert captured["input_ids"].tolist() == [[7, 41, 42, 7]]
    assert captured["mm_token_type_ids"].tolist() == [[0, 1, 2, 0]]
    assert captured["mm_token_type_ids"].dtype == captured["input_ids"].dtype
    assert inputs["mm_token_type_ids"].shape == (1, 5)
