"""The pipeline contract. Everything upstream (camera + WiLoR, replay, synthetic, later EMG) produces these types;
everything downstream (smoothing, the SOMA hand model, rendering) consumes only these. No Qt, no torch here.

Clock: every `t_ns` is on the host's monotonic clock (`time.monotonic_ns()`), stamped when the underlying sample
was *captured* (not when inference finished). Sources with their own clock (replay log times, an EMG band) map it
onto the monotonic clock themselves.

Hand model: NVIDIA's SOMA hand (py-soma-x `SOMAHandLayer`, 25 joints, see SOMA_JOINTS), with its own left and right
layers. Conventions, checked in tests/test_hand_model.py:
- Rotations are axis-angle (rotation vectors), relative to SOMA's T-pose (`absolute_pose=False`): all zeros is the
  rest hand, flat with the fingers along the model's -x (right hand). Each rotation is about axes of the model frame
  as they are in the rest pose, not about the joint's own bone axes.
- Left and right use the same numbers for the same gesture: the left layer's mesh for pose p is exactly the right
  layer's mesh for p reflected through the wrist (v -> -v). An EMG model can share one decoder across both arms.
- `wrist_position` is where the wrist joint (SOMA joint 0, the mesh origin) goes, unlike MANO's `transl`.
Sources that speak MANO (WiLoR, replayed mano.npz) convert in hand_viewer/mano/to_soma.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, runtime_checkable

import numpy as np

Side = Literal["left", "right"]
# SOMA hand skeleton (SOMAHandLayer docstring). Index 0 is the wrist; FINGER_JOINTS are rows 1..24, i.e. HandPose's
# finger_pose rows. Fingers other than the thumb have a metacarpal joint (…1) in the palm before the knuckle (…2).
SOMA_JOINTS = ("Wrist",
               "Thumb1", "Thumb2", "Thumb3", "ThumbEnd",
               "Index1", "Index2", "Index3", "Index4", "IndexEnd",
               "Middle1", "Middle2", "Middle3", "Middle4", "MiddleEnd",
               "Ring1", "Ring2", "Ring3", "Ring4", "RingEnd",
               "Pinky1", "Pinky2", "Pinky3", "Pinky4", "PinkyEnd")
FINGER_JOINTS = SOMA_JOINTS[1:]
NUM_FINGER_JOINTS = len(FINGER_JOINTS)   # 24
NUM_SHAPE = 20                           # SOMA hand identity PCA components
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
    """One hand at one instant. Only `finger_pose` is required: a source that only knows finger articulation (e.g. an
    EMG model, which predicts joint angles relative to the wrist) leaves the rest as None and the renderer places
    the hand at a fixed per-side anchor (wrist on the anchor)."""

    finger_pose: np.ndarray                   # (24, 3) axis-angle, FINGER_JOINTS order, T-pose relative; 0 = flat
    wrist_orient: np.ndarray | None = None    # (3,) axis-angle, rotates the whole hand in `frame`; None = unknown
    wrist_position: np.ndarray | None = None  # (3,) metres in `frame`; None = no position
    shape: np.ndarray | None = None           # (20,) SOMA hand identity coefficients; None = mean hand
    frame: PoseFrame = "camera"
    # 0..1, meaning is per source (WiLoR: detector score; EMG: model confidence). Downstream uses it only for
    # thresholding/display.
    confidence: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "finger_pose", _frozen(self.finger_pose, (NUM_FINGER_JOINTS, 3), "finger_pose"))
        object.__setattr__(self, "wrist_orient", _frozen(self.wrist_orient, (3,), "wrist_orient"))
        object.__setattr__(self, "wrist_position", _frozen(self.wrist_position, (3,), "wrist_position"))
        object.__setattr__(self, "shape", _frozen(self.shape, (NUM_SHAPE,), "shape"))


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
