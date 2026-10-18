"""The Denoise tab: Chroma Denoise moved out of Lab onto its own card and page, config unchanged."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.settings_catalog import DENOISE_FIELDS, FRAME_CARD_FIELDS, LAB_FIELDS, frame_card_rows
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.features.lab.models import LabConfig


def _panel(qapp) -> ControlsPanel:
    controller = MagicMock()
    controller.state = AppState()
    controller.active_diptych.return_value = None
    return ControlsPanel(controller)


def test_lab_and_denoise_split_the_lab_config_fields():
    every_field = set(LabConfig.__dataclass_fields__)
    assert set(LAB_FIELDS) | set(DENOISE_FIELDS) == every_field
    assert not set(LAB_FIELDS) & set(DENOISE_FIELDS)
    assert FRAME_CARD_FIELDS["denoise"] == DENOISE_FIELDS


def test_denoise_card_owns_exactly_the_denoise_rows():
    assert [r.label for r in frame_card_rows("denoise")] == ["Chroma Denoise", "Luma Denoise"]
    assert "Chroma Denoise" not in {r.label for r in frame_card_rows("lab")}


def test_slider_lives_on_the_denoise_sidebar_not_on_lab(qapp):
    panel = _panel(qapp)
    assert hasattr(panel.denoise_sidebar, "chroma_denoise_slider")
    assert not hasattr(panel.lab_sidebar, "chroma_denoise_slider")


def test_denoise_is_its_own_tab_between_look_and_finish(qapp):
    keys = [p["key"] for p in _panel(qapp).pages]
    assert keys.index("color") + 1 == keys.index("denoise") == keys.index("finish") - 1


def test_denoise_reset_leaves_lab_alone_and_lab_reset_leaves_denoise_alone(qapp):
    panel = _panel(qapp)
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(cfg, lab=replace(cfg.lab, chroma_denoise=2.0, sharpen=0.9))
    session = panel.controller.session

    panel._reset_lab_fields(DENOISE_FIELDS)
    lab = session.update_config.call_args.args[0].lab
    assert lab.chroma_denoise == 0.0 and lab.sharpen == 0.9

    panel._reset_lab_fields(LAB_FIELDS)
    lab = session.update_config.call_args.args[0].lab
    assert lab.chroma_denoise == 2.0 and lab.sharpen == LabConfig().sharpen
