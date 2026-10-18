import dataclasses
from unittest.mock import patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry import logic
from negpy.features.geometry.logic import is_pure_orientation, orient_rect, resolve_meter_rect
from negpy.services.rendering import image_processor


def _framed_image() -> np.ndarray:
    """A bright film box on a dark holder, off-centre, so a turn moves it."""
    img = np.full((300, 450, 3), 0.03, dtype=np.float32)
    rng = np.random.default_rng(1)
    img[60:230, 40:380] = 0.55 + 0.1 * rng.random((170, 340, 3), dtype=np.float32)
    return img


def test_orient_rect_matches_detection_on_the_turned_frame():
    img = _framed_image()
    geom = WorkspaceConfig().geometry
    upright = resolve_meter_rect(img, geom, 1600)
    assert upright is not None
    for rotation, flip_h, flip_v in (
        (1, False, False),
        (2, False, False),
        (3, False, False),
        (0, True, False),
        (0, False, True),
        (1, True, False),
    ):
        turned = dataclasses.replace(geom, rotation=rotation, flip_horizontal=flip_h, flip_vertical=flip_v)
        found = resolve_meter_rect(img, turned, 1600)
        assert found is not None
        predicted = orient_rect(upright, rotation, flip_h, flip_v)
        assert max(abs(a - b) for a, b in zip(found, predicted)) < 0.01, (rotation, flip_h, flip_v)


def test_four_quarter_turns_bring_the_rect_back():
    rect = (0.1, 0.2, 0.8, 0.9)
    assert np.allclose(orient_rect(rect, 4, False, False), rect)
    assert np.allclose(orient_rect(orient_rect(rect, 1, False, False), 3, False, False), rect)


def test_pure_orientation_excludes_fine_rotation_and_keystone():
    geom = WorkspaceConfig().geometry
    assert is_pure_orientation(geom)
    assert not is_pure_orientation(dataclasses.replace(geom, fine_rotation=1.0))
    assert not is_pure_orientation(dataclasses.replace(geom, converge_v=0.2))


def test_rotating_detects_only_once_per_frame():
    img = _framed_image()
    base = WorkspaceConfig()
    memo: dict = {}
    calls = []
    real = logic.resolve_meter_rect

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    with patch.object(image_processor, "resolve_meter_rect", counting):
        for rotation in (0, 1, 2, 3, 0, 2):
            settings = dataclasses.replace(base, geometry=dataclasses.replace(base.geometry, rotation=rotation))
            image_processor._resolve_meter_rect(img, settings, memo, "frame")

    assert len(calls) == 1
