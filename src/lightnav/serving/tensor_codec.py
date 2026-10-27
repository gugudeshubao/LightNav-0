"""Lossless FP16 byte transport. Requires system libzstd only for this codec."""
import ctypes
import ctypes.util
from functools import lru_cache


@lru_cache(maxsize=1)
def _library():
    path = ctypes.util.find_library("zstd")
    if not path:
        raise ValueError("zstd-shuffle requires system libzstd")
    lib = ctypes.CDLL(path)
    size, pointer = ctypes.c_size_t, ctypes.c_void_p
    for name, args, result in (
        ("ZSTD_compressBound", [size], size),
        ("ZSTD_compress", [pointer, size, pointer, size, ctypes.c_int], size),
        ("ZSTD_decompress", [pointer, size, pointer, size], size),
        ("ZSTD_findFrameCompressedSize", [pointer, size], size),
        ("ZSTD_isError", [size], ctypes.c_uint),
    ):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


def compress_fp16(raw: bytes) -> bytes:
    if not raw or len(raw) % 2:
        raise ValueError("expected FP16 bytes")
    lib = _library()
    shuffled = raw[0::2] + raw[1::2]
    output = ctypes.create_string_buffer(lib.ZSTD_compressBound(len(raw)))
    count = lib.ZSTD_compress(output, len(output), shuffled, len(shuffled), 3)
    if lib.ZSTD_isError(count):
        raise ValueError("zstd compression failed")
    return output.raw[:count]


def decompress_fp16(packed: bytes, expected_bytes: int) -> bytes:
    if not 0 < expected_bytes <= 6*1024*1024 or expected_bytes % 2:
        raise ValueError("invalid FP16 output size")
    lib = _library()
    frame_size = lib.ZSTD_findFrameCompressedSize(packed, len(packed))
    if lib.ZSTD_isError(frame_size) or frame_size != len(packed):
        raise ValueError("invalid or trailing zstd frame")
    output = ctypes.create_string_buffer(expected_bytes)
    count = lib.ZSTD_decompress(output, expected_bytes, packed, len(packed))
    if lib.ZSTD_isError(count) or count != expected_bytes:
        raise ValueError("invalid zstd output size")
    shuffled = output.raw
    raw = bytearray(expected_bytes)
    middle = expected_bytes // 2
    raw[0::2], raw[1::2] = shuffled[:middle], shuffled[middle:]
    return bytes(raw)
