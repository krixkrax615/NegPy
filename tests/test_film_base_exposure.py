import math

import tifffile
import numpy as np

from negpy.services.scanning.capture_info import PLUSTEK_DEFAULT_EXPOSURE, base_for_exposure, film_exposure


def _tif(tmp_path, name, description):
    p = tmp_path / name
    tifffile.imwrite(p, np.zeros((4, 4, 3), np.uint16), description=description)
    return str(p)


def test_metered_exposure_is_read_from_the_capture_line(tmp_path):
    p = _tif(tmp_path, "a.tif", "NegPy Plustek scan: 3600 dpi, 16-bit, auto-exposure 53895, driver capture sharpening")
    assert film_exposure(p) == 53895


def test_plustek_scan_without_metering_uses_the_model_default(tmp_path):
    p = _tif(tmp_path, "b.tif", "NegPy Plustek scan: 3600 dpi, 16-bit, driver capture sharpening")
    assert film_exposure(p) == PLUSTEK_DEFAULT_EXPOSURE


def test_foreign_file_has_no_exposure(tmp_path):
    assert film_exposure(_tif(tmp_path, "c.tif", "")) is None
    assert film_exposure(str(tmp_path / "missing.tif")) is None


def test_base_shifts_by_the_log_of_the_exposure_ratio():
    out = base_for_exposure((-0.30, -0.50, -0.80), 28000, 56000)
    shift = math.log10(2.0)
    assert all(abs(o - (b + shift)) < 1e-9 for o, b in zip(out, (-0.30, -0.50, -0.80)))


def test_base_is_untouched_when_an_exposure_is_unknown_or_equal():
    base = (-0.3, -0.5, -0.8)
    assert base_for_exposure(base, None, 56000) == base
    assert base_for_exposure(base, 56000, None) == base
    assert base_for_exposure(base, 56000, 56000) == base


