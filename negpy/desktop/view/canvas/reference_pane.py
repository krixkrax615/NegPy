from typing import Callable, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.canvas.overlay import draw_view_badge
from negpy.desktop.view.canvas.widget import apply_wheel_zoom_notches, clamp_canvas_zoom_level, wheel_notch_delta
from negpy.desktop.view.styles.templates import icon_button, tool_toggle
from negpy.desktop.view.styles.theme import THEME


class ReferencePane(QWidget):
    """A frame pinned beside the canvas, to match the rest of a roll against.

    Zoom is relative to fit, like the canvas: scroll zooms at the cursor, drag pans,
    double-click switches between fit and one image pixel per screen pixel."""

    closed = pyqtSignal()
    #: (zoom, pan x, pan y) after the user moves the view while linked; pan is the image shift as a fraction of the pane.
    view_changed = pyqtSignal(float, float, float)

    def __init__(self, background: Callable[[], QColor]):
        super().__init__()
        self._background = background
        self._image: Optional[QImage] = None
        self._name = ""
        self._zoom = 1.0
        self._center = QPointF()
        self._drag_from: Optional[QPointF] = None
        self.setMinimumWidth(160)
        self.close_btn = icon_button("fa5s.times", "Close the reference")
        self.close_btn.setParent(self)
        self.close_btn.clicked.connect(self.closed)
        self.link_btn = tool_toggle("fa5s.link", "", "Link the zoom and position of this view to the canvas")
        self.link_btn.setParent(self)
        self.link_btn.setChecked(True)
        self.link_btn.toggled.connect(lambda on: self._emit_view() if on else None)

    @property
    def image(self) -> Optional[QImage]:
        return self._image

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_reference(self, image: Optional[QImage], name: str = "") -> None:
        self._image = image
        self._name = name
        self._reset_view()
        self.update()

    @property
    def linked(self) -> bool:
        return self.link_btn.isChecked()

    def apply_view(self, zoom: float, pan_x: float, pan_y: float) -> None:
        """Take the canvas view without sending it back."""
        if not self._has_image():
            return
        self._zoom = max(1.0, zoom)
        scale = self._scale()
        self._center = QPointF(
            self._image.width() / 2 - pan_x * self.width() / scale,
            self._image.height() / 2 - pan_y * self.height() / scale,
        )
        self._clamp_center()
        self.update()

    def _emit_view(self) -> None:
        if not (self.linked and self._has_image()):
            return
        scale = self._scale()
        pan_x = (self._image.width() / 2 - self._center.x()) * scale / max(1, self.width())
        pan_y = (self._image.height() / 2 - self._center.y()) * scale / max(1, self.height())
        self.view_changed.emit(self._zoom, pan_x, pan_y)

    def _reset_view(self) -> None:
        self._zoom = 1.0
        if self._image is not None and not self._image.isNull():
            self._center = QPointF(self._image.width() / 2, self._image.height() / 2)

    def _area(self) -> QRectF:
        return QRectF(self.rect()).adjusted(THEME.space_2xl, THEME.space_2xl, -THEME.space_2xl, -THEME.space_2xl)

    def _scale(self) -> float:
        assert self._image is not None
        area = self._area()
        fit = min(area.width() / self._image.width(), area.height() / self._image.height())
        return fit * self._zoom

    def _clamp_center(self) -> None:
        """Keep the image filling the pane on an axis where it is larger, centered where it is smaller."""
        assert self._image is not None
        area, scale = self._area(), self._scale()
        coords = []
        for c, size, view in (
            (self._center.x(), self._image.width(), area.width()),
            (self._center.y(), self._image.height(), area.height()),
        ):
            half = view / scale / 2
            coords.append(size / 2 if half * 2 >= size else min(max(c, half), size - half))
        self._center = QPointF(*coords)

    def _to_image(self, pos: QPointF) -> QPointF:
        area, scale = self._area(), self._scale()
        return self._center + (pos - area.center()) / scale

    def _has_image(self) -> bool:
        return self._image is not None and not self._image.isNull()

    def _zoom_to(self, zoom: float, anchor: QPointF) -> None:
        """Zoom with the image point under ``anchor`` (widget coords) held still."""
        under = self._to_image(anchor)
        self._zoom = zoom
        self._center = under - (anchor - self._area().center()) / self._scale()
        self._clamp_center()
        self.update()
        self._emit_view()

    def wheelEvent(self, event) -> None:
        u = wheel_notch_delta(event)
        if not self._has_image() or u == 0.0:
            event.ignore()
            return
        self._zoom_to(max(1.0, apply_wheel_zoom_notches(self._zoom, u)), event.position())
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if not self._has_image() or event.button() != Qt.MouseButton.LeftButton:
            return
        if self._zoom > 1.0:
            self._reset_view()
            self.update()
            self._emit_view()
            return
        one_to_one = 1.0 / self._scale()
        self._zoom_to(max(1.0, clamp_canvas_zoom_level(one_to_one)), event.position())

    def mousePressEvent(self, event) -> None:
        if self._has_image() and event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.0:
            self._drag_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is None:
            return
        self._center -= (event.position() - self._drag_from) / self._scale()
        self._drag_from = event.position()
        self._clamp_center()
        self.update()
        self._emit_view()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
        self.unsetCursor()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.close_btn.move(self.width() - self.close_btn.width() - THEME.space_lg, THEME.space_lg)
        self.link_btn.move(self.close_btn.x() - self.link_btn.width() - THEME.space_lg, THEME.space_lg)
        if self._has_image():
            self._clamp_center()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._background())
        if self._image is None or self._image.isNull():
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        area, scale = self._area(), self._scale()
        top_left = area.center() - self._center * scale
        painter.setClipRect(area)
        painter.drawImage(QRectF(top_left.x(), top_left.y(), self._image.width() * scale, self._image.height() * scale), self._image)
        painter.setClipping(False)
        text = f"REFERENCE · {self._name}" if self._name else "REFERENCE"
        draw_view_badge(painter, text, THEME.space_xl, THEME.space_xl)
