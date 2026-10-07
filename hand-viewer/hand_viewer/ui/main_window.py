"""Main window: Encord header (source pickers, settings cog), video card | 3D hands card, footer stats; a ~60 Hz
timer drains the Session and redraws both panes. The cog hides the pickers, card headers and footer for a clean demo. All source/session lifecycle happens here, on the GUI thread."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from hand_viewer.app.session import Session, SessionState
from hand_viewer.core.hand_model import HandModel
from hand_viewer.core.smoothing import SmoothingConfig
from hand_viewer.core.types import VideoFrame, now_ns
from hand_viewer.sources import camera_permission
from hand_viewer.ui import theme
from hand_viewer.ui.hand_view import HandRenderer, PyqtgraphHandRenderer
from hand_viewer.ui.video_view import VideoView

EPISODES_DIR = Path(__file__).resolve().parents[3] / "encord-scene" / "out"
SOURCES = {"camera": "Camera + WiLoR", "replay": "Replay", "replay-wilor": "Replay video + WiLoR",
           "synthetic": "Synthetic"}
DEVICES = ("auto", "mps", "cuda", "cpu")
WILOR_RATE_HZ = 6.0     # what WiLoR manages on the M4 Pro; only used to pick smoothing settings
SYNTHETIC_RATE_HZ = 60.0
TICK_MS = 16


def find_episodes(root: Path = EPISODES_DIR) -> list[Path]:
    """Episode dirs with a mano.npz, newest first."""
    if not root.is_dir():
        return []
    eps = [d for d in root.iterdir() if (d / "mano.npz").is_file()]
    return sorted(eps, key=lambda d: (d / "mano.npz").stat().st_mtime, reverse=True)


def hfov_deg(frame: VideoFrame | None) -> float:
    k = frame.intrinsics if frame is not None else None
    return 60.0 if k is None else math.degrees(2 * math.atan(k.width / (2 * k.fx)))


class MainWindow(QMainWindow):
    # camera_permission.request answers on an arbitrary thread; this hops back to the GUI thread.
    _camera_access_answered = Signal(bool)

    def __init__(self, source: str = "camera", episode: Path | None = None, device: str = "auto",
                 camera: int | None = None):
        super().__init__()
        self.setWindowTitle("Encord hand viewer")
        self.setWindowIcon(theme.app_icon())
        self.resize(1500, 800)
        self.session: Session | None = None
        self.counts = {"video_frames": 0, "pose_frames": 0, "renders": 0, "errors": 0}
        self._last_status = ""
        self._last_error = ""
        self._smoothing_base = SmoothingConfig()
        self._camera_hfov: float | None = None   # the field of view the camera view was last set up with

        self.video = VideoView()
        self.renderer: HandRenderer = PyqtgraphHandRenderer()
        self.hand_model: HandModel | None = None
        try:
            self.hand_model = HandModel()
        except Exception as e:  # no SOMA model -> no 3D, but the video and stats still work
            self._last_error = f"SOMA hand model unavailable: {e}"

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_header(source, episode, device, camera))
        lay.addWidget(self._build_canvas(), 1)
        lay.addWidget(self._build_footer())
        self.setCentralWidget(root)
        self._set_view(self.view_box.currentIndex())

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)
        self._restart()

    # ------------------------------------------------------------------------------------------ widgets

    def _field(self, label: str, *widgets: QWidget) -> QWidget:
        """Eyebrow label + its controls, hidden and shown as one."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(theme.SPACE[2])
        h.addWidget(theme.eyebrow(label))
        for x in widgets:
            h.addWidget(x)
        return w

    def _build_header(self, source, episode, device, camera) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        h = QHBoxLayout(header)
        h.setContentsMargins(theme.SPACE[6], theme.SPACE[3], theme.SPACE[6], theme.SPACE[3])
        h.setSpacing(theme.SPACE[4])
        h.addWidget(theme.logo(22))
        h.addWidget(theme.divider())
        title = QLabel("Hand viewer")
        title.setFont(theme.display_font(16))
        h.addWidget(title)
        h.addStretch(1)
        # Everything the cog hides lives in `controls`; the logo, title and cog stay.
        self.controls = QWidget()
        outer, h = h, QHBoxLayout(self.controls)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(theme.SPACE[4])
        outer.addWidget(self.controls)

        self.source_box = QComboBox()
        for key, label in SOURCES.items():
            self.source_box.addItem(label, key)
        self.source_box.setCurrentIndex(max(0, self.source_box.findData(source)))
        h.addWidget(self._field("Source", self.source_box))

        self.camera_box = QComboBox()
        self.camera_box.setMinimumWidth(180)
        self._fill_cameras(camera)
        self.camera_field = self._field("Camera", self.camera_box)
        h.addWidget(self.camera_field)
        self.grant_button = QPushButton("Grant camera access")
        self.grant_button.setObjectName("primary")   # the one thing to do next, when it's shown
        self.grant_button.clicked.connect(self._grant_camera_access)
        h.addWidget(self.grant_button)
        self._camera_access_answered.connect(self._on_camera_access_answered)

        self.device_box = QComboBox()
        self.device_box.addItems(DEVICES)
        self.device_box.setCurrentText(device if device in DEVICES else "auto")
        self.device_field = self._field("Device", self.device_box)
        h.addWidget(self.device_field)

        self.episode_box = QComboBox()
        self.episode_box.setMinimumWidth(240)
        eps = find_episodes()
        if episode is not None and Path(episode).resolve() not in [e.resolve() for e in eps]:
            eps.insert(0, Path(episode))
        for ep in eps:
            self.episode_box.addItem(ep.name, str(ep))
        if episode is not None:
            self.episode_box.setCurrentIndex(self.episode_box.findData(str(Path(episode))))
        browse = QPushButton("…")
        browse.setObjectName("icon")
        browse.setToolTip("Choose an episode folder (containing mano.npz)")
        browse.clicked.connect(self._browse_episode)
        self.episode_field = self._field("Episode", self.episode_box, browse)
        h.addWidget(self.episode_field)

        self.source_box.currentIndexChanged.connect(self._restart)
        self.camera_box.currentIndexChanged.connect(self._restart)
        self.device_box.currentIndexChanged.connect(self._restart)
        self.episode_box.currentIndexChanged.connect(self._restart)

        self.settings_button = QPushButton()
        self.settings_button.setObjectName("icon")
        self.settings_button.setIcon(theme.icon("settings"))
        self.settings_button.setIconSize(QSize(18, 18))
        self.settings_button.setCheckable(True)
        self.settings_button.setChecked(True)
        self.settings_button.setToolTip("Show or hide the controls")
        self.settings_button.toggled.connect(self._set_controls_visible)
        outer.addWidget(self.settings_button)
        return header

    def _build_canvas(self) -> QWidget:
        canvas = QWidget()
        canvas.setObjectName("canvas")
        c = QHBoxLayout(canvas)
        c.setContentsMargins(theme.SPACE[4], theme.SPACE[4], theme.SPACE[4], theme.SPACE[4])

        self.video_card = theme.Card("Video", self.video)
        self.sync_check = theme.Switch("Sync to pose")
        self.sync_check.setToolTip("Show the video frame matching the (delayed, smoothed) pose instead of the latest")
        self.sync_check.toggled.connect(lambda on: self.session and self.session.set_sync_video(on))
        self.video_card.add_action(self.sync_check)

        self.hands_card = theme.Card("Hands", self.renderer.widget)
        self.hands_card.header_layout.insertWidget(2, theme.LegendDot("Left", theme.HAND_COLORS["left"]))
        self.hands_card.header_layout.insertWidget(3, theme.LegendDot("Right", theme.HAND_COLORS["right"]))
        self.smooth_check = theme.Switch("Smoothing")
        self.smooth_check.setChecked(True)
        self.smooth_check.toggled.connect(self._apply_smoothing)
        self.hands_card.add_action(self.smooth_check)
        self.view_box = QComboBox()
        self.view_box.addItems(["Orbit view", "Camera view"])
        self.view_box.setCurrentIndex(1)
        self.view_box.setToolTip("Camera view looks from the video camera, with its field of view")
        self.view_box.currentIndexChanged.connect(self._set_view)
        self.hands_card.add_action(self.view_box)
        heading = QPushButton("Zero heading")
        heading.setToolTip("World-frame poses (e.g. a wristband IMU): turn the hand to face away from you")
        heading.clicked.connect(self.renderer.zero_heading)
        self.hands_card.add_action(heading)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(theme.SPACE[4])
        splitter.addWidget(self.video_card)
        splitter.addWidget(self.hands_card)
        splitter.setSizes([750, 750])
        c.addWidget(splitter)
        return canvas

    def _build_footer(self) -> QWidget:
        footer = self.footer = QWidget()
        footer.setObjectName("footer")
        f = QHBoxLayout(footer)
        f.setContentsMargins(theme.SPACE[6], theme.SPACE[2], theme.SPACE[6], theme.SPACE[2])
        f.setSpacing(theme.SPACE[6])
        self.stats_label = QLabel()
        self.stats_label.setFont(theme.app_font())
        self.message_label = QLabel()
        self.message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addWidget(self.stats_label)
        f.addWidget(self.message_label, 1)
        return footer

    def _fill_cameras(self, index: int | None) -> None:
        try:
            from hand_viewer.sources.camera import list_cameras
            cams = list_cameras()
        except Exception:
            cams = []
        for dev_id, desc in cams:
            self.camera_box.addItem(desc, dev_id)
        if not cams:
            self.camera_box.addItem("No cameras found", None)
        if index is not None and 0 <= index < len(cams):
            self.camera_box.setCurrentIndex(index)

    def _update_toolbar_visibility(self, source: str) -> None:
        self.camera_field.setVisible(source == "camera")
        self._update_grant_button(source)
        self.device_field.setVisible(source in ("camera", "replay-wilor"))
        self.episode_field.setVisible(source in ("replay", "replay-wilor"))

    def _set_status(self, message: str, error: bool = False) -> None:
        self.message_label.setObjectName("error" if error else "")
        self.message_label.style().unpolish(self.message_label)
        self.message_label.style().polish(self.message_label)
        self.message_label.setText(message)

    # ------------------------------------------------------------------------------------------ camera access

    def _update_grant_button(self, source: str) -> None:
        status = camera_permission.status()
        self.grant_button.setVisible(source == "camera" and status != "granted")
        if status == "undetermined":
            self.grant_button.setText("Grant camera access")
            self.grant_button.setToolTip("Ask macOS for camera access (the prompt names your terminal / IDE)")
        else:
            self.grant_button.setText("Open camera settings")
            self.grant_button.setToolTip(f"Camera access is {status}; macOS won't ask again, so enable it in System "
                                         "Settings > Privacy & Security > Camera, then restart your terminal / IDE")

    def _grant_camera_access(self) -> None:
        if camera_permission.status() == "undetermined":
            self._set_status("Waiting for the macOS camera prompt…")
            camera_permission.request(self._camera_access_answered.emit)
        else:
            QDesktopServices.openUrl(QUrl(camera_permission.SETTINGS_URL))
            self._show_error("Enable camera access for your terminal / IDE in System Settings, then restart it "
                             "(macOS only applies the change to newly launched apps).")

    def _on_camera_access_answered(self, granted: bool) -> None:
        if granted:
            self._restart()
        else:
            self._update_grant_button(self.source_box.currentData())
            self._show_error("Camera access denied. Use \"Open camera settings\" to allow it.")

    def _browse_episode(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Episode directory", str(EPISODES_DIR))
        if not d:
            return
        i = self.episode_box.findData(d)
        if i < 0:
            self.episode_box.insertItem(0, Path(d).name, d)
            i = 0
        self.episode_box.setCurrentIndex(i)

    # ------------------------------------------------------------------------------------------ sessions

    def _build_session(self, source: str) -> tuple[Session, float]:
        """(session, expected pose rate in Hz). Heavy imports stay inside the branch that needs them."""
        if source == "camera":
            from hand_viewer.sources.camera import QtCameraSource
            from hand_viewer.sources.wilor import WilorSource   # torch + wilor_mini: only for the live source
            pose = WilorSource(self.device_box.currentText())
            return Session(pose, QtCameraSource(self.camera_box.currentData())), WILOR_RATE_HZ
        if source in ("replay", "replay-wilor"):
            from hand_viewer.sources.replay import Replay
            ep = self.episode_box.currentData()
            if not ep:
                raise FileNotFoundError(f"no episodes with mano.npz under {EPISODES_DIR}; use … to pick one")
            r = Replay(Path(ep))
            if source == "replay-wilor":
                # The recorded video through the live WiLoR path: tests camera mode without a camera.
                from hand_viewer.sources.wilor import WilorSource
                return Session(WilorSource(self.device_box.currentText()), r.frame_source), WILOR_RATE_HZ
            rate = 1e9 * r.clock.speed * r.num_rows / r.clock.period if r.num_rows else 15.0
            return Session(r.pose_source, r.frame_source), rate
        from hand_viewer.sources.synthetic import SyntheticSource
        return Session(SyntheticSource(rate_hz=SYNTHETIC_RATE_HZ), None), SYNTHETIC_RATE_HZ

    def _stop_session(self) -> None:
        if self.session is not None:
            self._set_status("Stopping…")
            self.message_label.repaint()
            self.session.stop()
            self.session = None

    def _restart(self) -> None:
        self._stop_session()
        source = self.source_box.currentData()
        self._update_toolbar_visibility(source)
        self.video.set_frame(None)
        self.video.set_placeholder("The synthetic source has no video; its hands animate on the right."
                                   if source == "synthetic" else "Waiting for video…")
        self.renderer.update(None, self.hand_model) if self.hand_model is not None else None
        self._last_status, self._last_error = "", ""
        if source == "camera" and (status := camera_permission.status()) != "granted":
            state = "hasn't been granted yet" if status == "undetermined" else f"is {status}"
            self.video.set_placeholder(f"Camera access {state}. Use \u201c{self.grant_button.text()}\u201d above to "
                                       "start the live feed.")
            self._set_status("")
            return
        try:
            session, rate = self._build_session(source)
        except Exception as e:
            self._show_error(f"{SOURCES[source]}: {type(e).__name__}: {e}")
            self.video.set_placeholder("The source didn't start. The message at the bottom says why.")
            return
        self._smoothing_base = SmoothingConfig.for_rate(rate) if hasattr(SmoothingConfig, "for_rate") \
            else SmoothingConfig()
        session.set_smoothing(replace(self._smoothing_base, enabled=self.smooth_check.isChecked()))
        session.set_sync_video(self.sync_check.isChecked())
        self.session = session
        session.start()

    def _apply_smoothing(self, on: bool) -> None:
        if self.session is not None:
            self.session.set_smoothing(replace(self._smoothing_base, enabled=on))

    def _set_controls_visible(self, on: bool) -> None:
        for w in (self.controls, self.video_card.header, self.hands_card.header, self.footer):
            w.setVisible(on)

    def _set_view(self, index: int) -> None:
        self._camera_hfov = None
        if index == 1:
            self._camera_hfov = hfov_deg(self.video.frame)
            self.renderer.set_camera_view(self._camera_hfov)
        else:
            self.renderer.reset_view()

    # ------------------------------------------------------------------------------------------ per tick

    def _tick(self) -> None:
        if self.session is None:
            return
        st = self.session.poll(now_ns())
        self._render(st)

    def _render(self, st: SessionState) -> None:
        if st.video is not None and st.video is not self.video.frame:
            self.video.set_frame(st.video)
            # The camera view starts before the first frame (60 degrees); match the real lens once it's known.
            if self._camera_hfov is not None and abs(hfov_deg(st.video) - self._camera_hfov) > 0.1:
                self._set_view(1)
        if self.hand_model is not None:
            self.renderer.update(st.hands, self.hand_model)
            self.counts["renders"] += st.hands is not None
        s = st.stats
        self.counts["video_frames"], self.counts["pose_frames"] = s.video_frames, s.pose_frames
        self.video_card.meta.setText(f"{s.video_fps:.0f} fps" if st.video is not None else "")
        self.hands_card.meta.setText(f"{s.pose_fps:.1f} fps" if s.pose_frames else "")
        latency = "–" if s.pose_latency_ms is None else f"{s.pose_latency_ms:.0f} ms"
        self.stats_label.setText(f"Video {s.video_fps:.1f} fps  ·  Pose {s.pose_fps:.1f} fps  ·  Latency {latency}")
        if st.status:
            self._last_status = st.status[-1]
        for e in st.errors:
            self._show_error(e)
        if not self._last_error:
            self._set_status(self._last_status)

    def _show_error(self, message: str) -> None:
        self.counts["errors"] += 1
        self._last_error = message
        print(f"error: {message}", flush=True)
        self._set_status(message, error=True)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._stop_session()
        super().closeEvent(event)


def warn_no_hand_model(parent: QWidget | None, window: MainWindow) -> None:
    if window.hand_model is None and window._last_error:
        QMessageBox.warning(parent, "Hand viewer", window._last_error)
