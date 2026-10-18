from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState


class _Canvas:
    def __init__(self, needs: bool) -> None:
        self.needs = needs

    def needs_full_res(self) -> bool:
        return self.needs


def _controller(*, hq: bool = True, active: bool = False, needs: bool = True, generation: int = 3):
    state = AppState(current_file_path="/roll/a.tif", hq_preview=hq, hq_active=active)
    ctl = SimpleNamespace(state=state, canvas=_Canvas(needs), _prefetch_gen=generation, load_file=MagicMock())
    return ctl


def test_full_res_needs_both_the_preference_and_an_active_upgrade():
    assert not AppState(hq_preview=True, hq_active=False).hq_full_res
    assert not AppState(hq_preview=False, hq_active=True).hq_full_res
    assert AppState(hq_preview=True, hq_active=True).hq_full_res


def test_zooming_past_the_preview_reloads_the_frame_at_full_resolution():
    ctl = _controller()

    AppController._maybe_upgrade_hq(ctl, 3)

    assert ctl.state.hq_active
    ctl.load_file.assert_called_once_with("/roll/a.tif", preserve_zoom=True)


def test_no_upgrade_while_the_preview_is_enough_or_hq_is_off_or_the_frame_changed():
    for kwargs, generation in (({"needs": False}, 3), ({"hq": False}, 3), ({"active": True}, 3), ({}, 2)):
        ctl = _controller(**kwargs)
        AppController._maybe_upgrade_hq(ctl, generation)
        ctl.load_file.assert_not_called()
