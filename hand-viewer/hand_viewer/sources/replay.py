"""Replay an offline episode written by encord-scene/mano_scene.py (frames/*.jpg + camera.json + mano.npz).

`Replay` exposes a FrameSource and a PoseSource that share one playback clock, so the VideoFrame and the HandFrame
made from the same image carry the same `t_ns` (the session's "sync video to pose" relies on that). Each source
runs its own worker thread and can be started/stopped independently; whichever starts first starts the clock.

mano.npz holds WiLoR's MANO parameters. The pose source converts every row to SOMA (hand_viewer/mano/to_soma.py) on
its worker before playing, which takes a few seconds; the video plays meanwhile, and poses join in step with it. The
result is cached under CACHE_DIR, keyed on the MANO arrays' contents, so later replays of the episode start at once.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hand_viewer.core.types import SIDES, CameraIntrinsics, FrameSink, HandFrame, HandPose, PoseSink, VideoFrame, now_ns

if TYPE_CHECKING:
    from hand_viewer.mano.to_soma import ManoToSoma

NAME = "replay"
DEFAULT_PERIOD_NS = 1_000_000_000 // 15  # gap between loops when the episode has a single row
CONVERT_CHUNK = 128                       # rows per MANO -> SOMA batch (~0.25 s)
CACHE_DIR: Path | None = Path.home() / ".cache" / "hand-viewer" / "soma"  # None: no cache
# Part of the cache key: bump whenever the conversion's output changes (to_soma.py, HandPose conventions).
CACHE_VERSION = 1
POSE_FIELDS = ("finger_pose", "wrist_orient", "wrist_position", "confidence")


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
    def __init__(self, clock: _Clock, mano: dict[str, np.ndarray], loop: bool, converter: ManoToSoma | None = None):
        super().__init__(clock, mano["log_time_ns"], loop)
        self._mano = mano
        self._converter = converter
        self._poses: dict[str, dict[int, HandPose]] | None = None

    def start(self, sink: PoseSink) -> None:
        super().start(sink)

    def _setup(self) -> None:
        if self._poses is None and self._converter is None:
            self._poses = load_cached_poses(self._mano)
        if self._poses is None:
            self._sink.status("Converting MANO to SOMA…")
            self.convert()
        if not self._stop.is_set():
            self._sink.status(f"Replaying {len(self._log_times)} rows")

    def convert(self) -> dict[str, dict[int, HandPose]]:
        """SOMA poses for every detected row, {side: {row: HandPose}}; computed once (batched per side). An injected
        converter bypasses the cache; the default one reads and writes it."""
        if self._poses is None and self._converter is None:
            self._poses = load_cached_poses(self._mano)
        if self._poses is None:
            from hand_viewer.mano.model import ManoParams
            from hand_viewer.mano.to_soma import ManoToSoma

            converter = self._converter or ManoToSoma()
            m, poses = self._mano, {}
            for side in SIDES:
                poses[side] = {}
                rows = np.flatnonzero(m[f"{side}_score"] > 0)  # mano_scene.py zeroes undetected rows, score 0
                for chunk in np.array_split(rows, max(1, len(rows) // CONVERT_CHUNK)):
                    if self._stop.is_set():  # chunked so stop() (which joins this thread) doesn't wait seconds
                        return {}
                    params = [ManoParams(m[f"{side}_hand_pose"][i], m[f"{side}_global_orient"][i],
                                         m[f"{side}_cam_t"][i], m[f"{side}_betas"][i]) for i in chunk]
                    converted = converter.convert(side, params, m[f"{side}_score"][chunk])
                    poses[side].update(zip(chunk.tolist(), converted))
            self._poses = poses
            if self._converter is None:
                save_cached_poses(self._mano, poses)
        return self._poses

    def hand_frame(self, i: int, t_ns: int) -> HandFrame:
        hands = {side: rows[i] for side, rows in self.convert().items() if i in rows}
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


def cache_path(mano: dict[str, np.ndarray]) -> Path | None:
    if CACHE_DIR is None:
        return None
    h = hashlib.sha1(f"v{CACHE_VERSION}".encode())
    for key in sorted(mano):
        h.update(key.encode())
        h.update(np.ascontiguousarray(mano[key]).tobytes())
    return CACHE_DIR / f"{h.hexdigest()}.npz"


def load_cached_poses(mano: dict[str, np.ndarray]) -> dict[str, dict[int, HandPose]] | None:
    """The poses `save_cached_poses` stored for these MANO arrays; None when absent or unreadable."""
    path = cache_path(mano)
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path) as npz:
            poses = {}
            for side in SIDES:
                rows = npz[f"{side}_rows"]
                fp, orient, pos, conf = (npz[f"{side}_{f}"] for f in POSE_FIELDS)
                poses[side] = {int(r): HandPose(fp[j], _none_if_nan(orient[j]), _none_if_nan(pos[j]), None, "camera",
                                                float(conf[j])) for j, r in enumerate(rows)}
            return poses
    except Exception:  # noqa: BLE001 - a bad cache file just means converting again
        return None


def save_cached_poses(mano: dict[str, np.ndarray], poses: dict[str, dict[int, HandPose]]) -> None:
    path = cache_path(mano)
    if path is None:
        return
    arrays = {}
    for side in SIDES:
        rows = sorted(poses[side])
        ps = [poses[side][r] for r in rows]
        arrays[f"{side}_rows"] = np.array(rows, np.int64)
        arrays[f"{side}_finger_pose"] = np.array([p.finger_pose for p in ps], np.float32).reshape(-1, 24, 3)
        arrays[f"{side}_wrist_orient"] = np.array([_nan_if_none(p.wrist_orient) for p in ps], np.float32).reshape(-1, 3)
        arrays[f"{side}_wrist_position"] = np.array([_nan_if_none(p.wrist_position) for p in ps],
                                                    np.float32).reshape(-1, 3)
        arrays[f"{side}_confidence"] = np.array([p.confidence for p in ps], np.float32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)  # atomic: a reader never sees half a file
    except OSError:
        pass  # read-only home etc.: replay still works, it just converts every time


def _nan_if_none(v):
    return np.full(3, np.nan) if v is None else v


def _none_if_nan(v):
    return None if np.isnan(v).any() else v


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
                 max_long_side: int | None = 960, converter: ManoToSoma | None = None):
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
        self.pose_source = ReplayPoseSource(self.clock, mano, loop, converter)
        self.frame_source = ReplayFrameSource(self.clock, frames, camera, loop, max_long_side, episode_dir)
