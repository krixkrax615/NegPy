import os
from functools import lru_cache

import tifffile

CAPTURE_PREFIX = "NegPy "


@lru_cache(maxsize=256)
def _read(path: str, mtime_ns: int) -> str:
    try:
        with tifffile.TiffFile(path) as tf:
            text = tf.pages[0].description or ""
    except Exception:
        return ""
    return text if text.startswith(CAPTURE_PREFIX) else ""


def read_capture_info(path: str) -> str:
    """The capture line a NegPy scan stored in its TIFF ImageDescription; empty for any other file."""
    if not path.lower().endswith((".tif", ".tiff")):
        return ""
    try:
        return _read(path, os.stat(path).st_mtime_ns)
    except OSError:
        return ""
