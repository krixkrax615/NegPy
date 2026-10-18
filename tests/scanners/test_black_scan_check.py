import numpy as np

from negpy.infrastructure.scanners.plustek_backend import scan_looks_black


def _frame(level: float, open_rows: int = 0) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = (rng.random((400, 600, 3)) * 0.1 + 0.9) * level
    if open_rows:
        img[:open_rows] = 65535.0
    return img.astype(np.uint16)


def test_normal_scan_is_fine():
    assert not scan_looks_black(_frame(50000.0, open_rows=20), expected_peak=0.85 * 65535)


def test_black_scan_with_open_light_is_flagged():
    assert scan_looks_black(_frame(900.0, open_rows=30), expected_peak=0.85 * 65535)


def test_dense_film_that_metering_also_saw_is_fine():
    assert not scan_looks_black(_frame(3000.0), expected_peak=3500.0)


def test_too_little_film_to_judge_is_fine():
    assert not scan_looks_black(np.full((400, 600, 3), 65535, dtype=np.uint16), expected_peak=0.85 * 65535)


def test_unknown_expectation_is_fine():
    assert not scan_looks_black(_frame(100.0), expected_peak=0.0)
