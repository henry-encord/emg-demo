"""FrameSource: a webcam via Qt Multimedia (QCamera -> QMediaCaptureSession -> QVideoSink).

Construct, start and stop on the GUI thread. QVideoSink.videoFrameChanged may fire on a multimedia thread; the
handler only converts and pushes into the (thread-safe) sink, under a lock that `stop()` also takes, so nothing is
emitted after `stop()` returns.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QCamera, QCameraDevice, QMediaCaptureSession, QMediaDevices, QVideoFrame, QVideoSink

from hand_viewer.core.types import CameraIntrinsics, FrameSink, VideoFrame, now_ns
from hand_viewer.sources import camera_permission

MAX_LONG_SIDE = 1280   # WiLoR downsizes to ~960 anyway; keeps conversion and the video blit cheap
NO_FRAMES_MS = 5000
PERMISSION_HINT = ("Use the toolbar's \"Grant camera access\" button; if access was denied before, enable your "
                   "terminal / IDE under System Settings > Privacy & Security > Camera and restart it.")


def list_cameras() -> list[tuple[bytes, str]]:
    """(device id, description) per camera, default camera first."""
    default = bytes(QMediaDevices.defaultVideoInput().id())
    cams = [(bytes(d.id()), d.description()) for d in QMediaDevices.videoInputs()]
    return sorted(cams, key=lambda c: c[0] != default)


def best_format(device: QCameraDevice):
    """Largest landscape format that runs at >= 24 fps. Qt's default on a MacBook camera is a square 1552x1552
    (Center Stage), which crops the sides of the scene where hands usually are."""
    formats = [f for f in device.videoFormats() if f.maxFrameRate() >= 24]
    landscape = [f for f in formats if f.resolution().width() > f.resolution().height()] or formats
    return max(landscape, key=lambda f: f.resolution().width() * f.resolution().height(), default=None)


def qimage_to_rgb(img: QImage) -> np.ndarray:
    """(H, W, 3) uint8 C-contiguous copy; rows in a QImage are padded to 4 bytes, so slice off bytesPerLine."""
    if img.format() != QImage.Format.Format_RGB888:
        img = img.convertToFormat(QImage.Format.Format_RGB888)
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
    return np.ascontiguousarray(buf[:, : w * 3].reshape(h, w, 3))


class QtCameraSource:
    name = "camera"

    def __init__(self, device_id: bytes | None = None, hfov_deg: float = 60.0):
        self.device_id = device_id
        self.hfov_deg = hfov_deg
        self._lock = threading.Lock()
        self._sink: FrameSink | None = None
        self._stopped = False
        self._got_frame = False
        self._camera: QCamera | None = None
        self._session: QMediaCaptureSession | None = None
        self._video_sink: QVideoSink | None = None
        self._ctx = QObject()   # context for the no-frames watchdog; deleting it cancels it
        self._intrinsics: CameraIntrinsics | None = None

    def start(self, sink: FrameSink) -> None:
        """Permission is asked by the UI (camera_permission.request) before a camera session is built; Qt's own
        QCameraPermission can't prompt from a bare python and reports Denied even when access is granted."""
        self._sink = sink
        if (status := camera_permission.status()) != "granted":
            sink.error(f"camera access is {status}. {PERMISSION_HINT}")
            return
        self._open()

    def _open(self) -> None:
        device = self._find_device()
        if device is None:
            self._sink.error("no camera found" if not QMediaDevices.videoInputs()
                             else f"camera {self.device_id!r} not found")
            return
        self._camera = QCamera(device)
        if (fmt := best_format(device)) is not None:
            self._camera.setCameraFormat(fmt)
        self._session = QMediaCaptureSession()
        self._video_sink = QVideoSink()
        self._session.setCamera(self._camera)
        self._session.setVideoSink(self._video_sink)
        self._video_sink.videoFrameChanged.connect(self._on_frame, Qt.ConnectionType.DirectConnection)
        self._camera.errorOccurred.connect(self._on_camera_error)
        self._sink.status(f"Opening {device.description()}…")
        self._camera.start()
        QTimer.singleShot(NO_FRAMES_MS, self._ctx, self._watchdog)

    def _find_device(self) -> QCameraDevice | None:
        devices = QMediaDevices.videoInputs()
        if self.device_id is None:
            d = QMediaDevices.defaultVideoInput()
            return None if d.isNull() else d
        return next((d for d in devices if bytes(d.id()) == self.device_id), None)

    def _on_camera_error(self, error, message: str) -> None:
        with self._lock:
            if not self._stopped:
                self._sink.error(f"camera error: {message or error}. {PERMISSION_HINT}")

    def _watchdog(self) -> None:
        with self._lock:
            if self._stopped or self._got_frame:
                return
            msg = f"no frames from the camera after {NO_FRAMES_MS // 1000} s; is another app using it?"
            # Granted but silent (camera busy in another app, still warming up): keep waiting, just say so.
            self._sink.status(msg)

    def _on_frame(self, frame: QVideoFrame) -> None:
        t = now_ns()   # capture stamp: the best we have from Qt without a device clock mapping
        if self._stopped or not frame.isValid():
            return
        img = frame.toImage()
        if img.isNull():
            return
        if max(img.width(), img.height()) > MAX_LONG_SIDE:
            img = img.scaled(MAX_LONG_SIDE, MAX_LONG_SIDE, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.FastTransformation)
        rgb = qimage_to_rgb(img)
        h, w = rgb.shape[:2]
        if self._intrinsics is None or (self._intrinsics.width, self._intrinsics.height) != (w, h):
            self._intrinsics = CameraIntrinsics.from_fov(w, h, self.hfov_deg)
        vf = VideoFrame(t_ns=t, image=rgb, intrinsics=self._intrinsics, meta={"device": self.device_id})
        with self._lock:
            if self._stopped:
                return
            if not self._got_frame:
                self._got_frame = True
                self._sink.status(f"Camera {w}×{h}")
            self._sink.video(vf)

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._ctx.deleteLater()   # drops a pending watchdog
        if self._video_sink is not None:
            try:
                self._video_sink.videoFrameChanged.disconnect(self._on_frame)
            except (RuntimeError, TypeError):
                pass
        if self._camera is not None:
            self._camera.stop()
        self._camera = self._session = self._video_sink = None
