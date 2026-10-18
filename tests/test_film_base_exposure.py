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


def test_exposure_shift_is_log10_over_the_default(tmp_path):
    from negpy.services.scanning.capture_info import exposure_shift

    p = _tif(tmp_path, "d.tif", "NegPy Plustek scan: 3600 dpi, 16-bit, auto-exposure 56000")
    assert abs(exposure_shift(p) - math.log10(56000 / PLUSTEK_DEFAULT_EXPOSURE)) < 1e-9
    assert exposure_shift(_tif(tmp_path, "e.tif", "")) == 0.0
    assert exposure_shift(None) == 0.0


def test_contact_sheet_reads_a_plustek_scans_exposure_in_stops(tmp_path):
    from negpy.services.export.contact_sheet_roll import read_scan_ev

    p = _tif(tmp_path, "f.tif", "NegPy Plustek scan: 3600 dpi, 16-bit, auto-exposure 56000")
    assert abs(read_scan_ev(p) - math.log2(56000 / PLUSTEK_DEFAULT_EXPOSURE)) < 1e-9


def test_a_late_loaded_frame_takes_the_roll_baseline_at_its_own_exposure():
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.assets import rolls
    from tests.test_film_base import _repo_with_roll

    repo, roll_id = _repo_with_roll()
    rolls.set_roll_normalization(repo, roll_id, (-3.0, -3.1, -3.2), (-0.1, -0.2, -0.3))
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, use_luma_average=True, use_color_average=True))

    out = rolls.resolve_roll_baseline(repo, roll_id, "hash-a", cfg, shift=0.3)

    assert out.process.locked_floors == tuple(v + 0.3 for v in (-3.0, -3.1, -3.2))
    assert out.process.locked_ceils == tuple(v + 0.3 for v in (-0.1, -0.2, -0.3))
