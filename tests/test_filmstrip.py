from unittest.mock import MagicMock

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QListView

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.shortcut_registry import REGISTRY
from negpy.desktop.view.sidebar.files import FileBrowser, FilmstripView
from negpy.infrastructure.storage.repository import StorageRepository


def _browser():
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.load_file_settings_many.return_value = {}
    repo.get_max_history_index.return_value = 0
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [{"name": f"{i}.tif", "path": f"/r/{i}.tif", "hash": f"h{i}"} for i in range(6)]
    session.asset_model.refresh()
    controller = MagicMock()
    controller.session = session
    controller.thumbnail_refresh_running = False
    return FileBrowser(controller)


def test_the_filmstrip_is_one_row_that_shares_the_strips_model_and_selection(qapp):
    browser = _browser()
    strip = browser.filmstrip_view

    assert isinstance(strip, FilmstripView)
    assert strip.flow() == QListView.Flow.LeftToRight and not strip.isWrapping()
    assert strip.model() is browser.list_view.model()
    assert strip.selectionModel() is browser.list_view.selectionModel()
    assert strip.height() < 2 * FilmstripView.CELL


def test_the_filmstrip_keeps_its_cell_size_when_resized_and_the_wheel_scrolls_sideways(qapp):
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent

    strip = _browser().filmstrip_view
    strip.resize(300, strip.height())
    strip.show()
    qapp.processEvents()

    assert strip.iconSize().width() == FilmstripView.CELL
    bar = strip.horizontalScrollBar()
    assert bar.maximum() > 0
    at = QPointF(50, 40)
    strip.wheelEvent(
        QWheelEvent(
            at, at, QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False
        )
    )
    assert bar.value() > 0


def test_a_click_on_the_filmstrip_opens_the_frame(qapp):
    browser = _browser()
    browser._activate_file = MagicMock()
    strip = browser.filmstrip_view
    strip.resize(900, strip.height())
    strip.show()
    qapp.processEvents()
    index = strip.model().index(2, 0)

    QTest.mouseClick(strip.viewport(), Qt.MouseButton.LeftButton, pos=strip.visualRect(index).center())

    browser._activate_file.assert_called_once()


def test_the_filmstrip_has_a_shortcut():
    assert REGISTRY["toggle_filmstrip"].default_key == "F6"
