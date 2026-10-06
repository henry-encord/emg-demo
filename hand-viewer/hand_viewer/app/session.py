"""The app session: owns one (PoseSource, FrameSource | None) pair for one run and is the sink both push into.

No Qt here. Sources call in from their own threads; the UI timer drains state with `poll()` on the GUI thread.
Each run gets a run id, and sources are handed a sink bound to it, so anything a source emits after `stop()` (or
after an error ended the run) is dropped instead of leaking into the next session.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from hand_viewer.core.smoothing import PoseSmoother, SmoothingConfig
from hand_viewer.core.types import FrameConsumer, FrameSource, HandFrame, PoseSource, VideoFrame, now_ns

VIDEO_BUFFER_NS = 1_000_000_000   # ring buffer span for "sync video to pose"
POSE_HISTORY_NS = 1_000_000_000   # recent HandFrames, replayed into a new smoother when smoothing settings change
EMA_ALPHA = 0.1
STALE_NS = 1_000_000_000          # a rate with no sample for this long (and > 3 intervals) reads as 0


class _Rate:
    """EMA of sample intervals -> rate in Hz, decaying to 0 when samples stop."""

    def __init__(self):
        self.interval_ns: float | None = None
        self.last_ns: int | None = None
        self.count = 0

    def tick(self, t_ns: int) -> None:
        if self.last_ns is not None and t_ns > self.last_ns:
            dt = t_ns - self.last_ns
            self.interval_ns = dt if self.interval_ns is None else (1 - EMA_ALPHA) * self.interval_ns + EMA_ALPHA * dt
        self.last_ns = t_ns
        self.count += 1

    def hz(self, now: int) -> float:
        if self.interval_ns is None or self.last_ns is None:
            return 0.0
        if now - self.last_ns > max(STALE_NS, 3 * self.interval_ns):
            return 0.0
        return 1e9 / self.interval_ns


@dataclass(frozen=True)
class SessionStats:
    video_fps: float = 0.0
    pose_fps: float = 0.0
    pose_latency_ms: float | None = None   # EMA of (wall time at emit - HandFrame.t_ns): capture-to-result latency
    video_frames: int = 0
    pose_frames: int = 0


@dataclass(frozen=True)
class SessionState:
    video: VideoFrame | None              # latest frame, or the one nearest the rendered pose in sync mode
    hands: HandFrame | None               # smoothed pose to render now
    stats: SessionStats
    status: list[str] = field(default_factory=list)   # new since the last poll
    errors: list[str] = field(default_factory=list)   # new since the last poll
    running: bool = False


class _BoundSink:
    """What sources actually get: forwards to the session only while its run is current."""

    def __init__(self, session: Session, run_id: int):
        self._s, self._run = session, run_id

    def pose(self, frame: HandFrame) -> None:
        self._s._on_pose(self._run, frame)

    def video(self, frame: VideoFrame) -> None:
        self._s._on_video(self._run, frame)

    def status(self, message: str) -> None:
        self._s._on_status(self._run, message)

    def error(self, message: str) -> None:
        self._s._on_error(self._run, message)


class Session:
    """Implements PoseSink + FrameSink. Calling `pose()`/`video()`/... directly feeds the current run."""

    def __init__(self, pose_source: PoseSource, frame_source: FrameSource | None = None,
                 smoothing: SmoothingConfig | None = None):
        self.pose_source = pose_source
        self.frame_source = frame_source
        self._lock = threading.Lock()
        self._run_id = 0
        self._running = False
        self._started = False
        self._stop_requested = False      # set from a worker thread on error; poll() does the actual stop
        self._sync_video = False
        self._smoothing = smoothing or SmoothingConfig()
        self._smoother = PoseSmoother(self._smoothing)
        self._video: deque[VideoFrame] = deque()
        self._poses: deque[HandFrame] = deque()
        self._pending: list[HandFrame] = []
        self._status: list[str] = []
        self._errors: list[str] = []
        self._video_rate = _Rate()
        self._pose_rate = _Rate()
        self._latency_ns: float | None = None
        self._consumer = pose_source if isinstance(pose_source, FrameConsumer) else None

    # ------------------------------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Pose source first, so a FrameConsumer is ready before the first frame is forwarded to it."""
        with self._lock:
            if self._started:
                raise RuntimeError("a Session runs at most once; build a new one")
            self._started = self._running = True
            self._run_id += 1
            sink = _BoundSink(self, self._run_id)
        try:
            self.pose_source.start(sink)
            if self.frame_source is not None:
                self.frame_source.start(sink)
        except Exception as e:  # a source that raises from start() is a bug, but don't take the UI down
            self._on_error(sink._run, f"{type(e).__name__}: {e}")
            self.stop()

    def stop(self) -> None:
        """Joins both sources (pose first). Call on the GUI thread (a Qt camera must be stopped there). Idempotent."""
        with self._lock:
            self._run_id += 1        # everything still in flight is now stale
            self._running = False
            self._stop_requested = False
        for src in (self.pose_source, self.frame_source):
            if src is None:
                continue
            try:
                src.stop()
            except Exception as e:
                with self._lock:
                    self._errors.append(f"stopping {getattr(src, 'name', src)}: {type(e).__name__}: {e}")

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    # ------------------------------------------------------------------------------------------ settings

    def set_sync_video(self, on: bool) -> None:
        with self._lock:
            self._sync_video = on

    def set_smoothing(self, config: SmoothingConfig) -> None:
        """New smoother, primed with the last second of poses so the hands don't vanish on toggle."""
        smoother = PoseSmoother(config)
        with self._lock:
            for f in self._poses:   # call on the UI thread (the smoother's only thread)
                smoother.push(f)
            self._smoothing, self._smoother = config, smoother
            self._pending = []   # already in _poses, so already replayed

    # ------------------------------------------------------------------------------------------ sinks

    def pose(self, frame: HandFrame) -> None:
        self._on_pose(self._run_id, frame)

    def video(self, frame: VideoFrame) -> None:
        self._on_video(self._run_id, frame)

    def status(self, message: str) -> None:
        self._on_status(self._run_id, message)

    def error(self, message: str) -> None:
        self._on_error(self._run_id, message)

    def _on_pose(self, run: int, frame: HandFrame) -> None:
        t = now_ns()
        with self._lock:
            if run != self._run_id or not self._running:
                return
            self._pose_rate.tick(t)
            lat = t - frame.t_ns
            self._latency_ns = lat if self._latency_ns is None else (1 - EMA_ALPHA) * self._latency_ns + EMA_ALPHA * lat
            self._pending.append(frame)   # PoseSmoother isn't thread-safe: poll() feeds it on the UI thread
            self._poses.append(frame)
            while self._poses and frame.t_ns - self._poses[0].t_ns > POSE_HISTORY_NS:
                self._poses.popleft()

    def _on_video(self, run: int, frame: VideoFrame) -> None:
        with self._lock:
            if run != self._run_id or not self._running:
                return
            self._video_rate.tick(frame.t_ns)
            self._video.append(frame)
            while len(self._video) > 1 and frame.t_ns - self._video[0].t_ns > VIDEO_BUFFER_NS:
                self._video.popleft()
            consumer = self._consumer
        # Outside the lock: submit_frame is non-blocking by contract, but it takes the consumer's own lock.
        if consumer is not None:
            consumer.submit_frame(frame)

    def _on_status(self, run: int, message: str) -> None:
        with self._lock:
            if run == self._run_id:
                self._status.append(message)

    def _on_error(self, run: int, message: str) -> None:
        """Fatal for the run. We may be on the source's own worker, which can't join itself, so poll() stops it."""
        with self._lock:
            if run != self._run_id or not self._running:
                return
            self._errors.append(message)
            self._running = False
            self._stop_requested = True
            self._run_id += 1

    # ------------------------------------------------------------------------------------------ UI side

    def poll(self, t_ns: int) -> SessionState:
        with self._lock:
            stop = self._stop_requested
            pending, self._pending = self._pending, []
            for f in pending:
                self._smoother.push(f)
            hands = self._smoother.sample(t_ns)
            video = self._pick_video(hands)
            status, self._status = self._status, []
            errors, self._errors = self._errors, []
            stats = SessionStats(
                video_fps=self._video_rate.hz(t_ns),
                pose_fps=self._pose_rate.hz(t_ns),
                pose_latency_ms=None if self._latency_ns is None else self._latency_ns / 1e6,
                video_frames=self._video_rate.count,
                pose_frames=self._pose_rate.count,
            )
            running = self._running
        if stop:
            self.stop()
            with self._lock:
                errors += self._errors   # anything stop() itself reported
                self._errors = []
        return SessionState(video=video, hands=hands, stats=stats, status=status, errors=errors, running=running)

    def _pick_video(self, hands: HandFrame | None) -> VideoFrame | None:
        if not self._video:
            return None
        if not self._sync_video or hands is None:
            return self._video[-1]
        return min(self._video, key=lambda f: abs(f.t_ns - hands.t_ns))
