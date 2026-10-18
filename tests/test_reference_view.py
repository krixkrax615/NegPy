from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QSplitter, QWidget

from negpy.desktop.view.canvas.reference_pane import ReferencePane
from negpy.desktop.view.main_window import MainWindow
from negpy.desktop.session import AppState
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE


def _window(metrics, path="/roll/frame_07.tif"):
    pane = ReferencePane(lambda: QColor("black"))
    splitter = QSplitter()
    splitter.addWidget(pane)
    splitter.addWidget(QWidget())
    splitter.resize(800, 400)
    splitter.show()
    pane.hide()
    win = SimpleNamespace(
        reference_pane=pane,
        central_splitter=splitter,
        state=AppState(current_file_path=path, last_metrics=metrics),
        controller=MagicMock(),
        canvas=MagicMock(),
    )
    win.controller.display_transform_params.return_value = (WORKING_COLOR_SPACE, None, None)
    win._reference_snapshot = lambda: MainWindow._reference_snapshot(win)
    win.close_reference = lambda: MainWindow.close_reference(win)
    win._sync_reference_view = lambda zoom: MainWindow._sync_reference_view(win, zoom)
    win.canvas.zoom_level = 1.0
    win.canvas.pan_offset = QPointF(0, 0)
    return win


def test_pinning_shows_the_frame_beside_the_canvas_and_a_second_press_closes_it(qapp):
    win = _window({"base_positive": np.full((30, 40, 3), 0.5, dtype=np.float32)})

    MainWindow.toggle_reference(win)
    assert win.reference_pane.isVisible()
    assert (win.reference_pane.image.width(), win.reference_pane.image.height()) == (40, 30)

    MainWindow.toggle_reference(win)
    assert not win.reference_pane.isVisible()
    assert win.reference_pane.image is None


def test_nothing_to_pin_without_a_rendered_frame(qapp):
    win = _window({}, path="")

    MainWindow.toggle_reference(win)

    assert not win.reference_pane.isVisible()
    win.canvas.hud.showMessage.assert_called_once()


def test_the_pane_paints_its_frame_and_closes_from_its_button(qapp):
    pane = ReferencePane(lambda: QColor("black"))
    image = QImage(40, 30, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    pane.set_reference(image, "frame_07.tif")
    pane.resize(300, 200)
    closed = []
    pane.closed.connect(lambda: closed.append(True))

    shot = pane.grab().toImage()
    pane.close_btn.click()

    assert shot.pixelColor(150, 100) == QColor("white")
    assert closed == [True]


def _pane_with_image():
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    pane = ReferencePane(lambda: QColor("black"))
    image = QImage(400, 300, QImage.Format.Format_RGB888)
    image.fill(QColor("white"))
    pane.set_reference(image, "frame_07.tif")
    pane.resize(300, 200)

    def wheel(notches: int, at: QPointF) -> None:
        event = QWheelEvent(
            at,
            at,
            QPoint(),
            QPoint(0, 120 * notches),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        pane.wheelEvent(event)

    return pane, wheel


def test_the_reference_zooms_at_the_cursor_and_back_to_fit(qapp):
    from PyQt6.QtCore import QPointF

    pane, wheel = _pane_with_image()
    at = QPointF(170, 110)
    under = pane._to_image(at)

    wheel(-8, at)

    assert pane.zoom > 1.0
    assert abs(pane._to_image(at).x() - under.x()) < 0.5 and abs(pane._to_image(at).y() - under.y()) < 0.5
    wheel(40, at)
    assert pane.zoom == 1.0


def test_a_drag_pans_the_zoomed_reference_and_a_new_pin_resets_the_view(qapp):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtTest import QTest

    pane, wheel = _pane_with_image()
    wheel(-5, QPointF(150, 100))
    before = QPointF(pane._center)

    QTest.mousePress(pane, Qt.MouseButton.LeftButton, pos=QPointF(150, 100).toPoint())
    QTest.mouseMove(pane, QPointF(120, 100).toPoint())
    QTest.mouseRelease(pane, Qt.MouseButton.LeftButton, pos=QPointF(120, 100).toPoint())

    assert pane._center.x() > before.x()
    pane.set_reference(pane.image, "frame_08.tif")
    assert pane.zoom == 1.0


def test_a_linked_reference_follows_the_canvas_and_reports_its_own_moves(qapp):
    from PyQt6.QtCore import QPointF

    pane, wheel = _pane_with_image()
    seen = []
    pane.view_changed.connect(lambda z, x, y: seen.append((z, x, y)))

    pane.apply_view(3.0, 0.1, -0.05)
    assert pane.zoom == 3.0 and not seen
    wheel(-1, QPointF(150, 100))
    assert seen and seen[-1][0] > 3.0

    seen.clear()
    pane.link_btn.setChecked(False)
    wheel(-1, QPointF(150, 100))
    assert not seen


def test_apply_view_and_emitted_view_round_trip(qapp):
    pane, _ = _pane_with_image()
    seen = []
    pane.view_changed.connect(lambda z, x, y: seen.append((z, x, y)))

    pane.apply_view(4.0, 0.2, 0.1)
    pane._emit_view()

    assert seen[-1] == (4.0, seen[-1][1], seen[-1][2])
    assert abs(seen[-1][1] - 0.2) < 1e-6 and abs(seen[-1][2] - 0.1) < 1e-6
