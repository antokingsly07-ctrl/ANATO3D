"""Dependency-free grayscale PNG rendering for slice thumbnails.

The interactive slice viewer receives *raw* pixel values (base64) so the client
can apply its own windowing.  Thumbnails, however, are rendered on the server
and shipped as PNG so the UI can show previews without decoding the full array.

Only 8-bit grayscale PNG is produced, which keeps the encoder tiny (zlib for
compression, struct for chunk headers) and avoids adding Pillow to the
dependency set.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from .preprocessing import apply_window, auto_window

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def to_uint8(array: np.ndarray, *, center: float | None = None, width: float | None = None) -> np.ndarray:
    """Map an intensity slice to 0..255 using a window/level."""
    data = np.asarray(array, dtype=np.float32)
    if center is not None and width is not None and width > 0:
        return apply_window(data, float(center), float(width))
    derived_center, derived_width = auto_window(data)
    return apply_window(data, derived_center, derived_width)


def downscale(array: np.ndarray, max_size: int) -> np.ndarray:
    """Nearest-neighbour decimation so the longest edge is <= ``max_size``."""
    if max_size <= 0:
        return array
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return array
    step = int(np.ceil(longest / float(max_size)))
    return array[::step, ::step]


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_png_gray(image: np.ndarray) -> bytes:
    """Encode a 2D uint8 array as an 8-bit grayscale PNG."""
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if array.ndim != 2:
        raise ValueError("encode_png_gray expects a 2D array")
    if array.size == 0:
        array = np.zeros((1, 1), dtype=np.uint8)

    height, width = int(array.shape[0]), int(array.shape[1])

    # Each scanline is prefixed with a filter-type byte (0 = None).
    raw = bytearray()
    for row in array:
        raw.append(0)
        raw.extend(row.tobytes())

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    payload = zlib.compress(bytes(raw), 6)
    return (
        _PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", payload)
        + _chunk(b"IEND", b"")
    )


def render_slice_png(
    array: np.ndarray,
    *,
    center: float | None = None,
    width: float | None = None,
    max_size: int = 256,
) -> bytes:
    """Window, downscale and PNG-encode a single 2D slice."""
    windowed = to_uint8(array, center=center, width=width)
    windowed = downscale(windowed, max_size)
    return encode_png_gray(windowed)


__all__ = [
    "downscale",
    "encode_png_gray",
    "render_slice_png",
    "to_uint8",
]
