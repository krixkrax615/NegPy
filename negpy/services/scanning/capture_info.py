import math
import os
import re
from functools import lru_cache

import tifffile

CAPTURE_PREFIX = "NegPy "
#: The Plustek model exposure a scan without ``auto-exposure N`` in its capture line was taken at.
PLUSTEK_DEFAULT_EXPOSURE = 14000
_AUTO_EXPOSURE = re.compile(r"auto-exposure (\d+)")


@lru_cache(maxsize=256)
def _read(path: str, mtime_ns: int) -> str:
    try:
        with tifffile.TiffFile(path) as tf:
            text = tf.pages[0].description or ""
    except Exception:
        return ""
    return text if text.startswith(CAPTURE_PREFIX) else ""


def film_exposure(path: str) -> int | None:
    """The sensor exposure a NegPy Plustek scan was taken at, from its capture line: the metered
    ``auto-exposure N``, else the model default. None for any file without a capture line, so a base
    measured on it is never rescaled."""
    info = read_capture_info(path)
    if not info.startswith("NegPy Plustek scan"):
        return None
    m = _AUTO_EXPOSURE.search(info)
    return int(m.group(1)) if m else PLUSTEK_DEFAULT_EXPOSURE


def base_for_exposure(base_log: tuple, base_exposure: int | None, frame_exposure: int | None) -> tuple:
    """A film base measured at ``base_exposure`` as it reads on a frame scanned at ``frame_exposure``.
    Sensor data is linear in exposure, so every channel's log10 transmittance shifts by log10(ratio).
    Unchanged when either exposure is unknown."""
    if not base_exposure or not frame_exposure or base_exposure == frame_exposure:
        return tuple(base_log)
    shift = math.log10(frame_exposure / base_exposure)
    return tuple(float(v) + shift for v in base_log)


def read_capture_info(path: str) -> str:
    """The capture line a NegPy scan stored in its TIFF ImageDescription; empty for any other file."""
    if not path.lower().endswith((".tif", ".tiff")):
        return ""
    try:
        return _read(path, os.stat(path).st_mtime_ns)
    except OSError:
        return ""
