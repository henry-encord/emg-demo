"""Replay an offline episode written by encord-scene/mano_scene.py (frames/*.jpg + camera.json + mano.npz).

`Replay` exposes a FrameSource and a PoseSource that share one playback clock, so the VideoFrame and the HandFrame
made from the same image carry the same `t_ns` (the session's "sync video to pose" relies on that). Each source
runs its own worker thread and can be started/stopped independently; whichever starts first starts the clock.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from hand_viewer.core.types import SIDES, CameraIntrinsics, FrameSink, HandFrame, HandPose, PoseSink, VideoFrame, now_ns

NAME = "replay"
DEFAULT_PERIOD_NS = 1_000_000_000 // 15  # gap between loops when the episode has a single row


class _Clock:
    """Maps episode log time onto the host monotonic clock. Loop `k` is shifted by `k * period` of log time, where
    `period` is the episode span plus one frame gap, so `t_ns` keeps increasing across loops."""

    def __init__(self, log_times: np.ndarray, speed: float):
        self.t0 = int(log_times[0])
        rel = log_times - self.t0
        gap = int(np.median(np.diff(rel))) if len(rel) > 1 else DEFAULT_PERIOD_NS
        self.period = int(rel[-1]) + max(gap, 1)
        self.speed = speed
        self._start: int | None = None
        self._lock = threading.Lock()

    def start(self) -> int:
        with self._lock:
            if self._start is None:
                self._start = now_ns()
            return self._start

    def mono(self, loop: int, log_time_ns: int) -> int:
        """Host time at which the item logged at `log_time_ns` is due on loop `loop`."""
        assert self._start is not None
        return self._start + round((loop * self.period + int(log_time_ns) - self.t0) / self.speed)

    def latest(self, rel: np.ndarray, now: int) -> int:
        """Global step (loop * len(rel) + index) of the latest item in `rel` (log times relative to t0) already due
        at `now`; -1 if none is due yet."""
        assert self._start is not None
        elapsed = int((now - self._start) * self.speed)
        if elapsed < 0:
            return -1
        loop, within = divmod(elapsed, self.period)
        i = int(np.searchsorted(rel, within, side="right")) - 1
        return loop * len(rel) + i  # i == -1 rolls back to the previous loop's last item


class _Player:
    """One worker thread stepping through `log_times` on the shared clock. Items already superseded when the
    worker gets to them are dropped (never drift behind the clock). Subclasses fill in `_prepare` / `_emit`."""

    def __init__(self, clock: _Clock, log_times: np.ndarray, loop: bool):
        self.name = NAME
        self._clock, self._log_times, self._loop = clock, log_times, loop
        self._rel = log_times - clock.t0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink = None

    def start(self, sink) -> None:
        if self._thread is not None:
            raise RuntimeError(f"{type(self).__name__} can only be started once")
        self._sink = sink
        self._clock.start()
        self._thread = threading.Thread(target=self._run, name=f"{type(self).__name__}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        t = self._thread
        if t is None:  # never started: nothing to stop, and a later start() still works
            return
        self._stop.set()
        if t is not threading.current_thread():
            t.join()

    def _run(self) -> None:
        try:
            self._setup()
            n, count = 0, len(self._log_times)
            while not self._stop.is_set():
                n = max(n, self._clock.latest(self._rel, now_ns()))
                if not self._loop and n >= count:
                    self._sink.status("Replay finished")
                    return
                loop, i = divmod(n, count)
                item = self._prepare(i)  # decode before sleeping: the wait hides the decode time
                due = self._clock.mono(loop, self._log_times[i])
                if self._stop.wait(max(0, due - now_ns()) / 1e9):
                    return
                self._emit(due, loop, i, item)
                n += 1
        except Exception as e:  # noqa: BLE001 - contract: report, never raise out of the worker
            if not self._stop.is_set():
                self._sink.error(f"replay: {type(e).__name__}: {e}")

    def _setup(self) -> None:
        pass

    def _prepare(self, i: int):
        return None

    def _emit(self, t_ns: int, loop: int, i: int, item) -> None:
        raise NotImplementedError


class ReplayPoseSource(_Player):
    def __init__(self, clock: _Clock, mano: dict[str, np.ndarray], loop: bool):
        super().__init__(clock, mano["log_time_ns"], loop)
        self._mano = mano

    def start(self, sink: PoseSink) -> None:
        super().start(sink)

    def hand_frame(self, i: int, t_ns: int) -> HandFrame:
        m, hands = self._mano, {}
        for side in SIDES:
            score = float(m[f"{side}_score"][i])
            if score > 0:  # mano_scene.py leaves undetected rows zeroed with score 0
                hands[side] = HandPose(
                    hand_pose=m[f"{side}_hand_pose"][i].reshape(15, 3), global_orient=m[f"{side}_global_orient"][i],
                    transl=m[f"{side}_cam_t"][i], betas=m[f"{side}_betas"][i], frame="camera", confidence=score)
        return HandFrame(t_ns=t_ns, hands=hands, source=NAME)

    def _emit(self, t_ns, loop, i, item) -> None:
        self._sink.pose(self.hand_frame(i, t_ns))


class ReplayFrameSource(_Player):
    def __init__(self, clock: _Clock, frames: list[Path] | None, camera: dict | None, loop: bool,
                 max_long_side: int | None, episode_dir: Path):
        log_times = np.array([int(p.stem) for p in frames or []], np.int64)
        super().__init__(clock, log_times if len(log_times) else np.array([clock.t0], np.int64), loop)
        self._frames, self._camera, self._max_long_side, self._dir = frames, camera, max_long_side, episode_dir

    def start(self, sink: FrameSink) -> None:
        super().start(sink)

    def _setup(self) -> None:
        if not self._frames:
            raise FileNotFoundError(f"no frames/*.jpg in {self._dir}; pose-only replay")
        import cv2  # noqa: F401 - fail early (on the worker) if OpenCV is missing

    def _prepare(self, i: int):
        return decode_frame(self._frames[i], self._camera, self._max_long_side)

    def _emit(self, t_ns, loop, i, item) -> None:
        image, intrinsics = item
        self._sink.video(VideoFrame(t_ns=t_ns, image=image, intrinsics=intrinsics,
                                    meta={"log_time_ns": int(self._log_times[i]), "index": i, "loop": loop}))


def decode_frame(path: Path, camera: dict, max_long_side: int | None) -> tuple[np.ndarray, CameraIntrinsics]:
    """JPEG -> RGB, downscaled so the long side is at most `max_long_side`, with intrinsics scaled to match."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not decode {path}")
    h, w = bgr.shape[:2]
    if (w, h) != (camera["width"], camera["height"]):
        raise ValueError(f"{path.name} is {w}x{h} but camera.json says {camera['width']}x{camera['height']}")
    if max_long_side and max(w, h) > max_long_side:
        s = max_long_side / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        nw, nh = w, h
    sx, sy = nw / w, nh / h
    k = camera["k"]
    # Pixel centres sit at integer coords, so the principal point scales about -0.5, not 0.
    intrinsics = CameraIntrinsics(nw, nh, k[0] * sx, k[4] * sy, (k[2] + 0.5) * sx - 0.5, (k[5] + 0.5) * sy - 0.5)
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    rgb.setflags(write=False)
    return rgb, intrinsics


MANO_KEYS = {"global_orient": 3, "hand_pose": 45, "betas": 10, "cam_t": 3, "score": None}


def load_mano(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (run encord-scene/mano_scene.py on the episode first)")
    with np.load(path) as npz:
        if "log_time_ns" not in npz.files:
            raise ValueError(f"{path}: missing 'log_time_ns'")
        t = npz["log_time_ns"].astype(np.int64)
        n = len(t)
        if n == 0:
            raise ValueError(f"{path}: no rows")
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"{path}: log_time_ns must be strictly increasing")
        out = {"log_time_ns": t}
        for side in SIDES:
            for key, width in MANO_KEYS.items():
                name = f"{side}_{key}"
                if name not in npz.files:
                    raise ValueError(f"{path}: missing '{name}'")
                shape = (n,) if width is None else (n, width)
                arr = np.asarray(npz[name], np.float32)
                if arr.shape != shape:
                    raise ValueError(f"{path}: '{name}' has shape {arr.shape}, expected {shape}")
                out[name] = arr
    return out


class Replay:
    """Plays encord-scene/out/<episode>/ in real time (scaled by `speed`), looping if `loop`. Only mano.npz is
    required; without frames/ the frame source reports an error when started and the pose source still works."""

    def __init__(self, episode_dir: Path | str, *, speed: float = 1.0, loop: bool = True,
                 max_long_side: int | None = 960):
        episode_dir = Path(episode_dir)
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"episode dir {episode_dir} does not exist")
        if not speed > 0:
            raise ValueError(f"speed must be > 0, got {speed}")
        mano = load_mano(episode_dir / "mano.npz")

        frames = camera = None
        frames_dir = episode_dir / "frames"
        if frames_dir.is_dir():
            frames = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            cam_path = episode_dir / "camera.json"
            if not cam_path.is_file():
                raise FileNotFoundError(f"{cam_path} not found (needed to play frames/)")
            camera = json.loads(cam_path.read_text())
            missing = {"width", "height", "k"} - camera.keys()
            if missing or len(camera["k"]) != 9:
                raise ValueError(f"{cam_path}: needs width, height and a 9-element k (missing {sorted(missing)})")
            t = mano["log_time_ns"]
            if frames and (int(frames[0].stem) < t[0] or int(frames[-1].stem) > t[-1]):
                raise ValueError(f"frames/ spans {frames[0].stem}..{frames[-1].stem}, outside mano.npz's "
                                 f"{t[0]}..{t[-1]}; the episode's outputs are out of step")

        self.episode_dir = episode_dir
        self.num_rows = len(mano["log_time_ns"])
        self.num_frames = len(frames or [])
        self.clock = _Clock(mano["log_time_ns"], speed)
        self.pose_source = ReplayPoseSource(self.clock, mano, loop)
        self.frame_source = ReplayFrameSource(self.clock, frames, camera, loop, max_long_side, episode_dir)
