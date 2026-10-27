import base64
import ctypes.util
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.inference.policies import NavigationPolicy
from lightnav.serving.preprocessed import CONTRACT, decode_preprocessed
from lightnav.serving.tensor_codec import compress_fp16, decompress_fp16


def wire(pixels):
    return dict(contract=CONTRACT, shape=list(pixels.shape), source_hw=[640, 720],
                compression="zlib", data_b64=base64.b64encode(zlib.compress(pixels.tobytes())).decode())


def policy():
    bundle = SimpleNamespace(video_size=(2, 4), slowfast_tiers=None)
    engine = SimpleNamespace(bundle=bundle, aspect_mode="stretch", new_vit_cache=lambda: None,
                             reset_episode_state=lambda: None)
    return NavigationPolicy(engine, num_history_frames=2)


def test_edge_tensor_is_buffered_without_preprocessing(monkeypatch):
    pixels = np.linspace(-1, 1, 24, dtype=np.float16).reshape(2, 4, 3)
    tensor, hw = decode_preprocessed(wire(pixels))
    p = policy()
    monkeypatch.setattr(p, "_convert_frame", lambda _: pytest.fail("cloud preprocessing called"))
    p.observe_preprocessed(tensor, hw)
    assert torch.equal(p._get_video_tensor()[0], tensor)
    with pytest.raises(ValueError, match="representation"):
        p.observe(np.zeros((2, 4, 3), dtype=np.uint8))
    p.reset()
    assert p._buffer_len == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.01])
def test_reject_values_before_history_mutation(bad):
    pixels = np.zeros((2, 4, 3), dtype=np.float16)
    pixels[0, 0, 0] = bad
    with pytest.raises(ValueError):
        decode_preprocessed(wire(pixels))


def test_reject_shape_and_compression_overflow():
    value = wire(np.zeros((2, 4, 3), dtype=np.float16))
    value["shape"] = [1, 1, 3]
    with pytest.raises(ValueError):
        decode_preprocessed(value)
    p = policy()
    with pytest.raises(ValueError):
        p.observe_preprocessed(torch.zeros((3, 3, 4), dtype=torch.float16), (640, 720))
    assert p._buffer_len == 0 and p._next_frame_id == 0


def test_reject_trailing_stream_and_wrong_precision():
    value = wire(np.zeros((2, 4, 3), dtype=np.float16))
    raw = base64.b64decode(value["data_b64"]) + zlib.compress(b"extra")
    value["data_b64"] = base64.b64encode(raw).decode()
    with pytest.raises(ValueError):
        decode_preprocessed(value)
    with pytest.raises(ValueError):
        policy().observe_preprocessed(torch.zeros((3, 2, 4)), (640, 720))


@pytest.mark.skipif(ctypes.util.find_library("zstd") is None, reason="optional libzstd absent")
def test_zstd_preserves_every_bit_pattern_and_bounds_output():
    raw = np.arange(65536, dtype="<u2").tobytes()
    packed = compress_fp16(raw)
    assert decompress_fp16(packed, len(raw)) == raw
    for data, size in ((packed[:-1], len(raw)), (packed+packed, len(raw)),
                       (packed, len(raw)-2), (packed, len(raw)+2), (packed, 8*1024*1024)):
        with pytest.raises(ValueError):
            decompress_fp16(data, size)
    pixels = np.linspace(-1, 1, 24, dtype="<f2").reshape(2, 4, 3)
    value = wire(pixels)
    value.update(compression="zstd-shuffle", data_b64=base64.b64encode(compress_fp16(pixels.tobytes())).decode())
    tensor, _ = decode_preprocessed(value)
    assert np.array_equal(tensor.permute(1, 2, 0).numpy(), pixels)
