"""Exact byte permutations for persistent Volta QPN weights (no requantization)."""
from __future__ import annotations

import numpy as np

from ..layouts import encoded_size


def pack_planes(codes: bytes, scales: bytes, format: str, shape: tuple[int, int]) -> tuple[bytes, bytes]:
    """Pack whole 32-row tiles from row-major codes and logical row-major scales.

    NVFP4 codes have adjacent K nibbles; scales are one byte per K16 group.
    FP8 codes are one byte each; scales are the original two BF16 bytes per row.
    """
    encoded_size("volta_qpn_v1", format, shape)
    n, k = shape
    rows = np.arange(32)
    lanes = rows // 8 * 4 + rows % 4 + rows % 8 // 4 * 16
    order = np.argsort(lanes)
    raw = np.frombuffer(codes, dtype=np.uint8)
    if format == "fp8_e4m3fn_row_bf16":
        if len(codes) != n * k or len(scales) != n * 2:
            raise ValueError("FP8 plane sizes differ from shape")
        tiled = raw.reshape(n // 32, 32, k // 128, 8, 16)
        packed = tiled[:, order].transpose(0, 2, 3, 1, 4)
        return packed.tobytes(), scales
    if len(codes) != n * k // 2 or len(scales) != n * k // 16:
        raise ValueError("NVFP4 plane sizes differ from shape")
    groups = raw.reshape(n, k // 16, 8)
    values = np.empty((n, k // 16, 16), dtype=np.uint8)
    values[:, :, 0::2] = groups & 15
    values[:, :, 1::2] = groups >> 4
    order_k = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
    values = values[:, :, order_k]
    packed = values[:, :, 0::2] | (values[:, :, 1::2] << 4)
    packed = packed.reshape(n // 32, 32, k // 16, 8)[:, order].transpose(0, 2, 1, 3)
    scale = np.frombuffer(scales, dtype=np.uint8).reshape(n // 32, 32, k // 16)
    return packed.tobytes(), scale[:, order].transpose(0, 2, 1).tobytes()


def unpack_block_scales(scales: bytes, shape: tuple[int, int]) -> bytes:
    """Decode only the M128x4 scale addressing; numeric scale words are unchanged."""
    n, k = shape
    if n % 128 or k % 64 or len(scales) != n * k // 16:
        raise ValueError("invalid M128x4 scale plane")
    a = np.frombuffer(scales, dtype=np.uint8).reshape(n // 128, k // 64, 32, 4, 4)
    return a.transpose(0, 3, 2, 1, 4).tobytes()


def unpack_planes(codes: bytes, scales: bytes, format: str, shape: tuple[int, int]) -> tuple[bytes, bytes]:
    """Return exact row-major code words and logical scale words from Volta fragments."""
    encoded_size("volta_qpn_v1", format, shape)
    n, k = shape
    rows = np.arange(32)
    lanes = rows // 8 * 4 + rows % 4 + rows % 8 // 4 * 16
    raw = np.frombuffer(codes, dtype=np.uint8)
    if format == "fp8_e4m3fn_row_bf16":
        if len(codes) != n * k or len(scales) != n * 2:
            raise ValueError("FP8 plane sizes differ from shape")
        tiled = raw.reshape(n // 32, k // 128, 8, 32, 16).transpose(0, 3, 1, 2, 4)
        return tiled[:, lanes].tobytes(), scales
    if len(codes) != n * k // 2 or len(scales) != n * k // 16:
        raise ValueError("NVFP4 plane sizes differ from shape")
    packed = raw.reshape(n // 32, k // 16, 32, 8).transpose(0, 2, 1, 3)[:, lanes]
    packed = packed.reshape(n, k // 16, 8)
    words = np.empty((n, k // 16, 16), dtype=np.uint8)
    order = np.array([0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15])
    words[:, :, order[0::2]] = packed & 15
    words[:, :, order[1::2]] = packed >> 4
    logical = words[:, :, 0::2] | (words[:, :, 1::2] << 4)
    scale = np.frombuffer(scales, dtype=np.uint8).reshape(n // 32, k // 16, 32)
    return logical.tobytes(), scale.transpose(0, 2, 1)[:, lanes].tobytes()
