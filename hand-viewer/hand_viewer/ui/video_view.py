"""Left pane: the latest VideoFrame, aspect-preserving, with an optional text overlay (fps etc.)."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from hand_viewer.core.types import VideoFrame
from hand_viewer.ui import theme


class VideoView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 120)
        self._frame: VideoFrame | None = None   # keeps the numpy buffer alive while the QImage points into it
        self._image: QImage | None = None
        self._overlay = ""
        self._placeholder = "no video"

    @property
    def frame(self) -> VideoFrame | None:
        return self._frame

    def set_frame(self, frame: VideoFrame | None) -> None:
        if frame is self._frame:
            return
        self._frame = frame
        if frame is None:
            self._image = None
        else:
            img = frame.image
            h, w = img.shape[:2]
            # Wraps the buffer without copying; valid as long as self._frame holds it.
            self._image = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_RGB888)
        self.update()

    def set_overlay(self, text: str) -> None:
        if text != self._overlay:
            self._overlay = text
            self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def image_rect(self) -> QRectF:
        """Where the image is drawn, in widget coordinates (for future overlays: boxes, projected mesh)."""
        if self._image is None:
            return QRectF()
        iw, ih = self._image.width(), self._image.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.VIEWPORT))
        if self._image is None:
            p.setPen(QColor(theme.MUTED_TEXT))
            p.setFont(theme.app_font())
            p.drawText(self.rect().adjusted(24, 0, -24, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self._placeholder)
        else:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(self.image_rect(), self._image)
        if self._overlay:
            f = QFont(theme.app_font())
            f.setPixelSize(12)
            p.setFont(f)
            box = p.fontMetrics().boundingRect(0, 0, 1000, 1000, 0, self._overlay).adjusted(-8, -4, 8, 4)
            box.moveTopLeft(self.rect().topLeft() + QPoint(8, 8))
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.FOREGROUND))
            p.drawRoundedRect(box, theme.RADIUS_MD, theme.RADIUS_MD)
            p.setPen(QColor(theme.BACKGROUND))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, self._overlay)
        p.end()
