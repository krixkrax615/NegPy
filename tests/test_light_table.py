from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QListView, QStackedWidget, QStyle, QStyleOptionViewItem, QToolButton, QWidget

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.keyboard_shortcuts import _context_cancel
from negpy.desktop.view.main_window import MainWindow
from negpy.desktop.view.sidebar.files import LIGHT_TABLE_WHITE, FileBrowser, _FilmDelegate, light_table_film, picture_in_thumbnail
from negpy.desktop.view.styles.theme import THEME
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.metadata.gear_models import FilmStock, GearLibrary
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.image.logic import prepare_thumbnail
from negpy.services.assets.gear import GearProfiles
from negpy.services.film_edge.geometry import PERF_ACROSS, PERF_FROM_EDGE, SheetFormat, perforation_centers


class _Dock:
    def __init__(self, visible=True, floating=False):
        self.visible = visible
        self.floating = floating

    def isVisible(self):
        return self.visible

    def isFloating(self):
        return self.floating

    def setVisible(self, visible):
        self.visible = visible


def _window():
    stack = QStackedWidget()
    stack.addWidget(QWidget())
    grid = QListView()
    stack.addWidget(grid)
    button = QToolButton()
    button.setCheckable(True)
    win = SimpleNamespace(
        central_stack=stack,
        drawer=_Dock(),
        session_dock=_Dock(),
        _panels_before_light_table=[],
        session_panel=SimpleNamespace(file_browser=SimpleNamespace(light_table_btn=button, light_table_view=grid)),
        controller=MagicMock(),
        state=SimpleNamespace(selected_file_idx=0),
    )
    win.controller.session.asset_model.actual_to_display.return_value = -1
    win.light_table_active = lambda: MainWindow.light_table_active(win)
    return win


def test_the_light_table_takes_the_canvas_place_and_gives_it_back(qapp):
    win = _window()

    MainWindow.set_light_table(win, True)
    assert win.central_stack.currentIndex() == 1
    assert win.session_panel.file_browser.light_table_btn.isChecked()
    assert not win.drawer.visible and not win.session_dock.visible, "the grid gets the whole window"

    MainWindow.set_light_table(win, False)
    assert win.central_stack.currentIndex() == 0
    assert not win.session_panel.file_browser.light_table_btn.isChecked()
    assert win.drawer.visible and win.session_dock.visible


def test_the_light_table_will_not_open_on_an_empty_grid(qapp):
    win = _window()
    win.controller.session.asset_model.rowCount.return_value = 0
    win.state.uploaded_files = [{"name": "a.tif"}]

    MainWindow.set_light_table(win, True)

    assert win.central_stack.currentIndex() == 0
    assert not win.session_panel.file_browser.light_table_btn.isChecked()
    assert win.drawer.visible and win.session_dock.visible
    win.controller.set_status.assert_called_once_with("No frames match the filter", 3000)


def test_the_light_table_closes_when_its_grid_empties(qapp):
    win = _window()
    MainWindow.set_light_table(win, True)
    win.set_light_table = lambda on: MainWindow.set_light_table(win, on)
    win.controller.session.asset_model.rowCount.return_value = 0
    win.state.uploaded_files = [{"name": "a.tif"}]

    MainWindow._leave_empty_light_table(win)

    assert win.central_stack.currentIndex() == 0
    assert win.drawer.visible and win.session_dock.visible


def test_leaving_the_light_table_keeps_a_hidden_controls_panel_hidden(qapp):
    win = _window()
    win.drawer.visible = False

    MainWindow.set_light_table(win, True)
    MainWindow.set_light_table(win, False)

    assert not win.drawer.visible
    assert win.session_dock.visible


def test_escape_closes_the_light_table_before_anything_else(qapp):
    win = _window()
    MainWindow.set_light_table(win, True)
    win.set_light_table = lambda on: MainWindow.set_light_table(win, on)
    controller = MagicMock()

    _context_cancel(controller, win)

    assert not win.light_table_active()
    controller.cancel_active_tool.assert_not_called()


def test_one_key_hides_both_panels_then_brings_both_back(qapp):
    win = _window()

    MainWindow.toggle_side_panels(win)
    assert not win.drawer.visible and not win.session_dock.visible
    MainWindow.toggle_side_panels(win)
    assert win.drawer.visible and win.session_dock.visible
    saved = win.controller.session.repo.save_global_setting.call_args_list
    assert saved[-2:] == [(("panel_left_visible", True),), (("panel_right_visible", True),)]


@pytest.fixture
def browser(qapp):
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.load_file_settings_many.return_value = {}
    repo.get_max_history_index.return_value = 0
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [{"name": f"{i}.tif", "path": f"/r/{i}.tif", "hash": f"h{i}"} for i in range(3)]
    session.asset_model.refresh()
    controller = MagicMock()
    controller.session = session
    controller.thumbnail_refresh_running = False
    return FileBrowser(controller)


def test_the_light_table_shares_the_strip_selection(browser):
    assert browser.light_table_view.model() is browser.list_view.model()
    assert browser.light_table_view.selectionModel() is browser.list_view.selectionModel()


def test_opening_a_frame_from_the_light_table_asks_for_the_canvas(browser, monkeypatch):
    opened, left = [], []
    monkeypatch.setattr(browser, "_activate_file", opened.append)
    browser.light_table_opened.connect(lambda: left.append(True))
    index = browser.light_table_view.model().index(1, 0)

    browser._open_from_light_table(index)

    assert opened == [index] and left == [True]


def _qimage(img: Image.Image) -> QImage:
    arr = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    return QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_RGB888).copy()


@pytest.mark.parametrize("size", [(300, 200), (200, 300)])
def test_a_padded_thumbnail_gives_back_its_picture(qapp, size):
    thumb = prepare_thumbnail(Image.new("RGB", size, (200, 120, 40)), 120)

    rect = picture_in_thumbnail(_qimage(thumb))

    assert (rect.width(), rect.height()) == ((120, 80) if size[0] > size[1] else (80, 120))


def _hp5(**meta) -> tuple[WorkspaceConfig, GearLibrary]:
    config = WorkspaceConfig()
    config = replace(config, metadata=replace(config.metadata, film_stock_id="film-hp5-400", **meta))
    return config, GearLibrary(film_stocks=GearProfiles._read_bundled_list("film_stocks.json", FilmStock))


def test_the_light_table_film_is_the_rolls_stock_on_white(qapp):
    config, library = _hp5()

    geometry, style, palette = light_table_film({}, config, library, half_frame=False)

    assert geometry.format == SheetFormat.FULL_FRAME
    assert (style.maker, style.text) == ("ILFORD", "HP5 PLUS")
    assert palette.no_film == QColor(LIGHT_TABLE_WHITE).getRgb()[:3]
    assert light_table_film({}, config, library, half_frame=True)[0].format == SheetFormat.HALF_FRAME
    assert light_table_film({}, _hp5(format="120")[0], library, half_frame=False)[0].format == SheetFormat.MEDIUM


def test_the_gear_library_is_read_again_only_when_the_film_changes(browser):
    loads = browser.controller.load_gear_library
    loads.reset_mock()
    loads.return_value = _hp5()[1]
    browser._light_table_key = None

    browser.sync_ui()
    browser.sync_ui()
    assert loads.call_count == 1

    state = browser.session.state
    state.config = replace(state.config, metadata=replace(state.config.metadata, film_stock_id="film-hp5-400"))
    browser.sync_ui()
    assert loads.call_count == 2


def test_a_light_table_cell_is_one_frame_of_film(browser):
    view, delegate = browser.light_table_view, browser._light_table_delegate
    browser._light_table_key = None
    browser.sync_ui()
    view.resize(900, 600)
    view._relayout()
    cell = view.gridSize()
    s = cell.width() / 38.0
    assert cell.height() == round(35.0 * s) + view.STRIP_GAP

    canvas = QImage(cell.width(), cell.height(), QImage.Format.Format_RGB888)
    canvas.fill(QColor(LIGHT_TABLE_WHITE))
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, cell.width(), cell.height())
    painter = QPainter(canvas)
    delegate.paint(painter, option, view.model().index(0, 0))
    painter.end()

    top = (cell.height() - round(35.0 * s)) // 2
    hole = canvas.pixelColor(round(perforation_centers(0.0, 38.0)[0] * s), top + round((PERF_FROM_EDGE + PERF_ACROSS / 2) * s))
    rebate = canvas.pixelColor(cell.width() // 2, top + round(4.5 * s))
    assert hole == QColor(LIGHT_TABLE_WHITE)
    assert max(rebate.red(), rebate.green(), rebate.blue()) < 40


@pytest.mark.parametrize(("rotation", "top_lies_left"), [(0, False), (3, True)])
def test_a_portrait_frame_lies_on_the_strip_as_on_the_film(qapp, rotation, top_lies_left):
    upright = Image.new("RGB", (200, 300), (0, 0, 200))
    upright.paste((220, 0, 0), (0, 0, 200, 150))
    icon = QIcon(QPixmap.fromImage(_qimage(prepare_thumbnail(upright, 120))))
    delegate = _FilmDelegate(frame_geometry=lambda asset: GeometryConfig(rotation=rotation))
    delegate.set_film(*light_table_film({}, *_hp5(), half_frame=False))
    model = QStandardItemModel()
    model.appendRow(QStandardItem())

    picture = delegate._thumbnail(icon, model.index(0, 0)).toImage()

    assert picture.width() > picture.height()
    assert (picture.pixelColor(5, picture.height() // 2).red() > 150) == top_lies_left


@pytest.mark.parametrize("hover", [False, True])
def test_a_light_table_frame_is_never_dimmed_and_hover_rings_it_red(qapp, hover):
    delegate = _FilmDelegate()
    delegate.set_film(*light_table_film({}, *_hp5(), half_frame=False))
    model = QStandardItemModel()
    item = QStandardItem()
    item.setData(
        QIcon(QPixmap.fromImage(_qimage(prepare_thumbnail(Image.new("RGB", (300, 200), (40, 160, 220)), 120)))),
        Qt.ItemDataRole.DecorationRole,
    )
    model.appendRow(item)
    canvas = QImage(380, 350, QImage.Format.Format_RGB888)
    canvas.fill(QColor(LIGHT_TABLE_WHITE))
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 380, 350)
    if hover:
        option.state |= QStyle.StateFlag.State_MouseOver
    painter = QPainter(canvas)
    delegate.paint(painter, option, model.index(0, 0))
    painter.end()

    _top, window = delegate._layout(delegate._film[0], option.rect)
    assert canvas.pixelColor(window.center()) == QColor(40, 160, 220)
    ring = canvas.pixelColor(window.center().x(), window.top() - delegate._SELECTION_OUTSET + 1)
    assert (ring == QColor(THEME.accent_primary)) == hover
