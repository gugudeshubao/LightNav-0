"""Bounded transport for edge-preprocessed pixels; no image preprocessing here."""
import base64
import zlib

import numpy as np
import torch

from lightnav.serving.tensor_codec import decompress_fp16

CONTRACT = "lightnav.rgb-normalized.hwc-fp16.v1"
MAX_ELEMENTS = 3 * 1024 * 1024


def decode_preprocessed(value):
    if not isinstance(value, dict) or set(value) != {
        "contract", "shape", "source_hw", "compression", "data_b64"
    }:
        raise ValueError("invalid preprocessed frame fields")
    if value["contract"] != CONTRACT or value["compression"] not in ("zlib", "zstd-shuffle"):
        raise ValueError("unsupported preprocessed frame contract")
    shape, source = value["shape"], value["source_hw"]
    if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
            or any(type(v) is not int or not 1 <= v <= 4096 for v in shape)
            or not isinstance(source, list) or len(source) != 2
            or any(type(v) is not int or not 1 <= v <= 4096 for v in source)):
        raise ValueError("invalid frame dimensions")
    count = shape[0] * shape[1] * shape[2]
    payload = value["data_b64"]
    if count > MAX_ELEMENTS or not isinstance(payload, str) or len(payload) > 8*1024*1024:
        raise ValueError("preprocessed frame exceeds limit")
    packed = base64.b64decode(payload, validate=True)
    if value["compression"] == "zstd-shuffle":
        raw = decompress_fp16(packed, count*2)
    else:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(packed, count*2+1)
        if (len(raw) != count*2 or not decoder.eof or decoder.unused_data
                or decoder.unconsumed_tail):
            raise ValueError("invalid preprocessed frame byte count or stream")
    pixels = np.frombuffer(raw, dtype="<f2").reshape(shape).copy()
    if not np.isfinite(pixels).all() or (np.abs(pixels) > 1).any():
        raise ValueError("preprocessed pixels must be finite in [-1, 1]")
    return torch.from_numpy(pixels).permute(2, 0, 1), tuple(source)
