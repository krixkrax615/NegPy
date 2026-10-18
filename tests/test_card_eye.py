"""The per-card eye: a transient view of the edit with cards at their defaults, never part of the edit."""

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.card_eye import EYE_CARDS, card_edited, disabled_key, effective_config
from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.features.lab.models import LabConfig
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.rendering.lens import lens_decode_token, metadata_lens_corrections

_BASE = DEFAULT_WORKSPACE_CONFIG


def _edited():
    exp = replace(_BASE.exposure, density=1.7, grade=3.1, paper_black=0.2)
    lab = replace(_BASE.lab, saturation=1.4, chroma_denoise=2.0, luma_denoise=0.5)
    return replace(_BASE, exposure=exp, lab=lab)


def test_nothing_disabled_returns_the_same_config_object():
    cfg = _edited()
    assert effective_config(cfg, frozenset()) is cfg


def test_a_disabled_card_shows_its_own_fields_at_defaults_and_keeps_the_rest():
    cfg = _edited()
    out = effective_config(cfg, frozenset({"tone"}))
    assert out.exposure.density == _BASE.exposure.density
    assert out.exposure.grade == _BASE.exposure.grade
    assert out.exposure.paper_black == cfg.exposure.paper_black
    assert out.lab == cfg.lab


def test_lab_and_denoise_share_a_section_but_switch_separately():
    cfg = _edited()
    lab_off = effective_config(cfg, frozenset({"lab"}))
    assert lab_off.lab.saturation == LabConfig().saturation
    assert lab_off.lab.chroma_denoise == 2.0 and lab_off.lab.luma_denoise == 0.5
    den_off = effective_config(cfg, frozenset({"denoise"}))
    assert den_off.lab.chroma_denoise == 0.0 and den_off.lab.luma_denoise == 0.0
    assert den_off.lab.saturation == 1.4


def test_a_whole_section_card_resets_every_field():
    from negpy.features.toning.models import ToningConfig

    cfg = replace(_BASE, toning=replace(ToningConfig(), selenium_strength=0.8))
    assert effective_config(cfg, frozenset({"toning"})).toning == ToningConfig()


def test_the_source_config_is_never_mutated():
    cfg = _edited()
    snapshot = cfg.to_dict()
    effective_config(cfg, frozenset(EYE_CARDS))
    assert cfg.to_dict() == snapshot


def test_geometry_has_no_eye():
    assert "geometry" not in EYE_CARDS


def test_card_edited_sees_only_the_cards_own_fields():
    a = _edited()
    b = replace(a, lab=replace(a.lab, chroma_denoise=3.0))
    assert card_edited(a, b, "denoise")
    assert not card_edited(a, b, "lab")
    assert not card_edited(a, b, "tone")


def test_disabled_key_is_stable_whatever_the_order():
    assert disabled_key(frozenset({"paper", "tone"})) == disabled_key(frozenset({"tone", "paper"}))


def _controller(cfg, disabled=frozenset()):
    state = AppState(current_file_path="/roll/a.tif", current_file_hash="h1", config=cfg, disabled_cards=disabled)
    state.preview_raw = np.zeros((8, 8, 3), np.float32)
    state.preview_lens_token = lens_decode_token(metadata_lens_corrections(cfg), cfg.flatfield)
    ctl = MagicMock(spec=AppController)
    ctl.state = state
    ctl._render_debounce = MagicMock()
    ctl._is_rendering = False
    ctl._pending_render_task = None
    ctl._foreground_preview_generation = None
    ctl._decoded_source_token = None
    ctl._thumb_config = None
    ctl._eye_prev = None
    ctl.active_diptych.return_value = None
    ctl._effective_cam_xyz.return_value = (None, None)
    ctl._preview_gain_slices.return_value = ()
    ctl.roll_film_base.return_value = None
    ctl._render_memo_key.side_effect = lambda config=None: AppController._render_memo_key(ctl, config)
    ctl._render_state.side_effect = lambda: AppController._render_state(ctl)
    ctl.proof_profiles.return_value = ()
    return ctl


def _task(ctl):
    AppController.request_render(ctl)
    return ctl.render_requested.emit.call_args.args[0]


def test_the_preview_renders_the_effective_config_as_an_override_with_no_memo_or_thumbnail():
    ctl = _controller(_edited(), frozenset({"tone"}))
    task = _task(ctl)
    assert task.config.exposure.density == _BASE.exposure.density
    assert task.config_override
    assert task.memo_key == ""
    assert not task.wants_thumbnail
    assert ctl.state.config.exposure.density == 1.7


def test_with_every_eye_on_the_render_is_the_plain_one():
    cfg = _edited()
    ctl = _controller(cfg)
    task = _task(ctl)
    assert task.config is cfg
    assert not task.config_override
    assert task.memo_key != ""


def test_memo_keys_never_mix_eye_on_and_off():
    cfg = _edited()
    ctl = _controller(cfg)
    on = AppController._render_memo_key(ctl)
    ctl.state.disabled_cards = frozenset({"tone"})
    off = AppController._render_memo_key(ctl)
    ctl.state.disabled_cards = frozenset({"tone", "paper"})
    both = AppController._render_memo_key(ctl)
    assert len({on, off, both}) == 3


def test_toggle_flips_the_card_signals_and_renders():
    ctl = _controller(_edited())
    AppController.toggle_card_eye(ctl, "tone")
    assert ctl.state.disabled_cards == frozenset({"tone"})
    ctl.eye_changed.emit.assert_called_once()
    ctl.request_render.assert_called_once()
    AppController.toggle_card_eye(ctl, "tone")
    assert ctl.state.disabled_cards == frozenset()
    AppController.toggle_card_eye(ctl, "geometry")
    assert ctl.state.disabled_cards == frozenset()


def test_editing_a_card_that_is_off_turns_its_eye_back_on():
    cfg = _edited()
    ctl = _controller(cfg, frozenset({"tone", "paper"}))
    AppController._reenable_eyes_on_edit(ctl)
    ctl.state.config = replace(cfg, exposure=replace(cfg.exposure, density=2.2))
    AppController._reenable_eyes_on_edit(ctl)
    assert ctl.state.disabled_cards == frozenset({"paper"})


def test_switching_frames_keeps_the_eyes_and_edits_to_other_cards_leave_them():
    cfg = _edited()
    ctl = _controller(cfg, frozenset({"tone"}))
    AppController._reenable_eyes_on_edit(ctl)
    ctl.state.current_file_hash = "h2"
    ctl.state.config = replace(cfg, exposure=replace(cfg.exposure, density=0.4))
    AppController._reenable_eyes_on_edit(ctl)
    assert ctl.state.disabled_cards == frozenset({"tone"})
    ctl.state.config = replace(ctl.state.config, lab=replace(ctl.state.config.lab, saturation=0.9))
    AppController._reenable_eyes_on_edit(ctl)
    assert ctl.state.disabled_cards == frozenset({"tone"})


def test_the_eye_state_is_not_part_of_the_saved_edit():
    cfg = _edited()
    assert "disabled_cards" not in cfg.to_dict()
    assert AppState().disabled_cards == frozenset()


def test_only_the_preview_render_reads_the_eye():
    """Export, thumbnails, roll apply, copy/paste and history all read the saved config, never the eye view."""
    import inspect
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "negpy"
    users = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py") if "effective_config(" in p.read_text(encoding="utf-8"))
    assert users == ["desktop/card_eye.py", "desktop/controller.py"]
    src = inspect.getsource(AppController)
    assert src.count("effective_config(") == 1
    assert "effective_config(" in inspect.getsource(AppController.request_render)


def test_cards_with_an_eye_show_the_button_and_geometry_does_not(qapp):
    from negpy.desktop.view.sidebar.controls_panel import ControlsPanel

    controller = MagicMock()
    controller.state = AppState()
    controller.active_diptych.return_value = None
    controller.card_eye_off.return_value = False
    panel = ControlsPanel(controller)
    for card in EYE_CARDS:
        assert getattr(panel, f"{card}_section").eye_btn is not None
    assert panel.geometry_section.eye_btn is None

    panel.tone_section.eye_btn.click()
    controller.toggle_card_eye.assert_called_once_with("tone")

    controller.card_eye_off.side_effect = lambda c: c == "tone"
    panel._sync_eyes()
    assert not panel.tone_section.title_label.isEnabled()
    assert panel.paper_section.title_label.isEnabled()
