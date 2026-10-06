"""The pipeline contract. Everything upstream (camera + WiLoR, replay, synthetic, later EMG) produces these types;
everything downstream (smoothing, MANO, rendering) consumes only these. No Qt, no torch here.

Clock: every `t_ns` is on the host's monotonic clock (`time.monotonic_ns()`), stamped when the underlying sample
was *captured* (not when inference finished). Sources with their own clock (replay log times, an EMG band) map it
onto the monotonic clock themselves.

MANO conventions (validated against WiLoR output in encord-scene/out/*/mano.npz to ~1e-7 m):
- Rotations are axis-angle (rotation vectors). `hand_pose` zeros is a flat open hand (MANO without the pose mean,
  i.e. `smplx.MANOLayer` / `smplx.MANO(use_pca=False, flat_hand_mean=True)`).
- Only the right-hand MANO model is used. Right-hand params go straight in. Left-hand params use WiLoR's mirrored
  convention: negate the y and z components of `global_orient` and every `hand_pose` row, run the right-hand model,
  then negate vertex x (and reverse face winding). Translation is added after mirroring.
- `transl` is added to the MANO output vertices (like WiLoR's cam_t / smplx's transl). It is NOT the wrist position:
  MANO's root joint sits ~10 cm from the MANO origin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, runtime_checkable

import numpy as np

Side = Literal["left", "right"]
SIDES: tuple[Side, Side] = ("left", "right")

# "camera": OpenCV camera frame of the video (x right, y down, z forward, metres).
# "world": a gravity-aligned frame, z up (against gravity), metres, arbitrary heading (e.g. a wristband IMU); the
#          renderer offers "zero heading".
PoseFrame = Literal["camera", "world"]


def now_ns() -> int:
    return time.monotonic_ns()


def _frozen(a, shape, name) -> np.ndarray | None:
    if a is None:
        return None
    arr = np.array(a, dtype=np.float32).reshape(shape)  # copies
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class HandPose:
    """One hand at one instant. Only `hand_pose` is required: a source that only knows finger articulation (e.g. an
    EMG model, which predicts joint angles relative to the wrist) leaves the rest as None and the renderer places
    the hand at a fixed per-side anchor (root joint on the anchor)."""

    hand_pose: np.ndarray                    # (15, 3) axis-angle per finger joint, MANO joint order; 0 = flat
    global_orient: np.ndarray | None = None  # (3,) axis-angle wrist rotation in `frame`; None = source has none
    transl: np.ndarray | None = None         # (3,) metres in `frame`, added to MANO vertices; None = no position
    betas: np.ndarray | None = None          # (10,) MANO shape; None = mean hand
    frame: PoseFrame = "camera"
    # 0..1, meaning is per source (WiLoR: detector score; EMG: model confidence). Downstream uses it only for
    # thresholding/display.
    confidence: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "hand_pose", _frozen(self.hand_pose, (15, 3), "hand_pose"))
        object.__setattr__(self, "global_orient", _frozen(self.global_orient, (3,), "global_orient"))
        object.__setattr__(self, "transl", _frozen(self.transl, (3,), "transl"))
        object.__setattr__(self, "betas", _frozen(self.betas, (10,), "betas"))


@dataclass(frozen=True)
class HandFrame:
    """All hands a pose source saw at one instant. A side missing from `hands` means that hand isn't present."""

    t_ns: int
    hands: Mapping[Side, HandPose]
    source: str


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 60.0) -> CameraIntrinsics:
        """Guess for an uncalibrated webcam. Only affects absolute depth, not finger pose."""
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        return cls(width, height, float(f), float(f), width / 2, height / 2)


@dataclass(frozen=True)
class VideoFrame:
    t_ns: int
    image: np.ndarray                   # (H, W, 3) uint8 RGB, C-contiguous; treat as read-only
    intrinsics: CameraIntrinsics | None = None
    meta: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------- sources


@runtime_checkable
class PoseSink(Protocol):
    """Implemented by the app session. Every method is thread-safe and non-blocking; call from any thread."""

    def pose(self, frame: HandFrame) -> None: ...
    def status(self, message: str) -> None: ...   # e.g. "loading WiLoR…", "running on mps"
    def error(self, message: str) -> None: ...     # fatal for this source; the session stops it


@runtime_checkable
class FrameSink(Protocol):
    """Implemented by the app session. Thread-safe and non-blocking."""

    def video(self, frame: VideoFrame) -> None: ...
    def status(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...


@runtime_checkable
class PoseSource(Protocol):
    """Produces HandFrames.

    Lifecycle:
    - `start(sink)` returns immediately; slow setup (model loading) happens on the source's own worker thread and
      is reported via `sink.status`. Failures go to `sink.error`, never raised out of the worker.
    - `stop()` blocks until the worker has exited; no sink calls may happen after it returns. Idempotent.
    - A source instance is started at most once.
    """

    name: str

    def start(self, sink: PoseSink) -> None: ...
    def stop(self) -> None: ...


@runtime_checkable
class FrameSource(Protocol):
    """Produces VideoFrames for the video pane (and for any FrameConsumer pose source). Same lifecycle as
    PoseSource."""

    name: str

    def start(self, sink: FrameSink) -> None: ...
    def stop(self) -> None: ...


@runtime_checkable
class FrameConsumer(Protocol):
    """A pose source that needs video (e.g. WiLoR). The session forwards every VideoFrame here. Must be
    non-blocking: keep only the latest frame and drop older ones."""

    def submit_frame(self, frame: VideoFrame) -> None: ...
