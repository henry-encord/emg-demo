# Live MANO hand viewer (desktop) — design

Date: 2026-10-06
Status: building (revised after two reviews, 2026-10-06)

## Goal

A PyQt desktop app that shows a live video feed on the left and a 3D render of the hands in it on the right,
updating as the hands move. It does live what `encord-scene/mano_scene.py` does offline: WiLoR → MANO →
mesh.

Hard requirement: **the pose data source must be swappable.** v1 gets hand pose from a camera via WiLoR.
Later it will come from an EMG wristband that outputs joint angles. Everything after the source boundary
(MANO forward pass, smoothing, rendering) must not know or care where the pose came from.

New code only, in `hand-viewer/`. Nothing is imported from `encord-scene/`. The few useful bits (WiLoR
setup, the MPS conv workaround, the principal-point shift, the left-hand winding flip) are re-implemented here.

## What we learned (research, 2026-10-06)

Measured on the M4 Pro, MPS, WiLoR-mini, over 35 frames of the Glue episode with 2 hands in view:

| Input             | YOLO detect | WiLoR pose (2 hands) | Throughput |
| ----------------- | ----------- | -------------------- | ---------- |
| 1920×1200         | 18 ms       | 194 ms               | 4.7 fps    |
| 960×600 (½ scale) | 17 ms       | 140 ms               | 6.4 fps    |
| MANO forward, CPU, 2 hands |    | 0.27 ms              | —          |

- **WiLoR's ViT is the bottleneck.** Expect about 5–7 fps of pose updates on the Mac, while video plays at the
  camera's full rate. On a CUDA box (L4) it should be several times faster. So the design treats pose
  as arriving slower than video, at its own pace. The renderer interpolates or smooths, and never blocks
  the video.
- **The MANO forward pass is effectively free** (well under 1 ms), so the pipeline boundary can be *MANO
  parameters* rather than vertices. Each source sends parameters only, and the app turns them into a mesh
  in one place. That's what makes the EMG swap clean: an EMG model will never produce vertices.
- WiLoR's MANO layer (`smplx.MANOLayer`, rotation-matrix input, **no pose mean added**) means
  `hand_pose = 0` (axis-angle) is a **flat open hand**. That's a convenient zero for joint angles.
- WiLoR has only the right-hand MANO model. It gets left hands by mirroring: it flips the crop, negates the y/z
  of the axis-angle params, and negates the x of the vertices, which flips the triangle winding. We keep that
  convention (see contract below).
- The MPS conv contiguity hook from `mano_scene.py` is still needed. MPS isn't visible inside the Claude Code
  sandbox, which doesn't matter for the app.

## Architecture

```
 ┌──────────────┐  VideoFrame   ┌──────────────────┐
 │ FrameSource  │──────────────▶│  Video pane (L)  │
 │ camera/file  │               └──────────────────┘
 └──────┬───────┘
        │ VideoFrame (latest-only)
        ▼
 ┌──────────────┐                                    ════════ BREAKPOINT ════════
 │ PoseSource   │   HandFrame (MANO params per side)  ┌────────────┐  ┌──────────┐  ┌────────────┐
 │  WilorSource │────────────────────────────────────▶│ Smoother   │─▶│ MANO fwd │─▶│ 3D pane (R)│
 │  ReplaySource│                                     │ (optional) │  │ verts    │  └────────────┘
 │  SyntheticSrc│                                     └────────────┘  └──────────┘
 │  EmgSource   │  ← later: joint angles → JointAngleAdapter → HandFrame
 └──────────────┘
```

Two independent inputs:

- **`FrameSource`** produces images for the left pane: a live camera (OpenCV `VideoCapture`), a video file,
  or none. It's optional. With EMG we'll probably still want the camera next to the render for comparison,
  so the video feed and the pose source are deliberately decoupled.
- **`PoseSource`** produces `HandFrame`s. A source can subscribe to a `FrameSource` (WiLoR does) or ignore it
  (EMG, replay, synthetic).

### The contract (`hand_viewer/core/types.py`)

The code is the source of truth; this summarises it. Revised after review (see "Review changes" below).

- `HandPose`: `hand_pose (15,3)` is required. `global_orient`, `transl` and `betas` are optional (None = the
  source doesn't know it). `frame` is `"camera"` (OpenCV camera frame of the video) or `"world"` (gravity-aligned,
  arbitrary heading, e.g. a band IMU). `confidence` is 0..1. Arrays are copied and made read-only.
- `HandFrame(t_ns, hands: {side: HandPose}, source)`. `VideoFrame(t_ns, image RGB uint8, intrinsics, meta)`.
- **Clock:** every `t_ns` is `time.monotonic_ns()` at *capture*. Sources map their own clocks onto it.
- **Left hand:** negate y/z of `global_orient` and every `hand_pose` row, run the right-hand MANO, negate vertex x,
  reverse face winding, then add `transl`. This is WiLoR's convention, verified to about 1e-7 m against `mano.npz`.
- `transl` is added to the vertices; it is not the wrist position. A hand with `transl=None` is placed with its
  MANO root joint (J0) on a per-side anchor.
- Sources push into a thread-safe `PoseSink`/`FrameSink` (`pose|video`, `status`, `error`). `start()` doesn't
  block, and heavy setup runs on the source's worker. `stop()` joins, and nothing is emitted after it returns.
  A pose source that needs video implements `FrameConsumer.submit_frame` (latest-wins), and the session forwards
  frames to it.

Everything downstream of the sink only ever sees `HandFrame`. **Swapping to EMG is a new `PoseSource` class plus a
menu entry. Nothing else changes.**

### Module interfaces (v1)

```python
# core/mano.py (CPU torch, smplx MANOLayer, MANO_RIGHT.pkl from ~/.cache/wilor-mini/pretrained_models)
@dataclass
class HandMesh: vertices: np.ndarray  # (778,3) float32
                joints: np.ndarray    # (21,3) OpenPose order (MANO 16 + 5 tips), same frame as vertices
class ManoModel:
    def __init__(self, model_path: Path | None = None): ...
    def faces(self, side: Side) -> np.ndarray            # (1538,3) int32, outward-facing winding for that side
    def forward(self, side: Side, pose: HandPose, *, default_orient: np.ndarray | None = None,
                anchor: np.ndarray | None = None) -> HandMesh
        # global_orient None -> default_orient (or identity). transl None and anchor given -> root joint on anchor.

# core/smoothing.py
@dataclass
class SmoothingConfig: enabled: bool = True; min_cutoff: float = 1.0; beta: float = 0.3; d_cutoff: float = 1.0
                       render_delay_ms: float | None = None   # None = auto (about one measured pose interval)
                       hold_ms: float = 300                   # keep a lost hand this long, then drop it
                       freeze_betas_after: int = 10           # median of first N frames per hand; 0 = never
class PoseSmoother:
    def __init__(self, config: SmoothingConfig): ...
    def push(self, frame: HandFrame) -> None                 # every sample, any rate
    def sample(self, t_ns: int) -> HandFrame | None          # pose to render at wall time t_ns (after delay)
    def reset(self) -> None

# sources/replay.py
class Replay:          # plays encord-scene/out/<episode>/ (mano.npz + frames/*.jpg + camera.json) on one clock
    def __init__(self, episode_dir: Path, *, speed: float = 1.0, loop: bool = True): ...
    frame_source: FrameSource
    pose_source: PoseSource
# sources/synthetic.py
class SyntheticSource(PoseSource): def __init__(self, rate_hz: float = 60.0, sides=SIDES): ...
# sources/wilor.py
class WilorSource(PoseSource, FrameConsumer):
    def __init__(self, device: str = "auto", *, input_long_side: int = 960, fp16: bool | None = None,
                 det_conf: float = 0.3): ...
# sources/camera.py (Qt; construct on the GUI thread)
class QtCameraSource(FrameSource): def __init__(self, device_id: bytes | None = None, hfov_deg: float = 60.0): ...
# app/session.py
class Session(PoseSink, FrameSink):   # owns (FrameSource | None, PoseSource), run id, buffers, stats
    def __init__(self, pose_source, frame_source=None, smoothing: SmoothingConfig | None = None): ...
    def start(self) -> None; def stop(self) -> None
    def poll(self, t_ns: int) -> SessionState   # called by the UI timer (about 60 Hz): latest video frame,
                                               # smoothed HandFrame, stats, status/errors
```

### Sources

| Source            | v1? | Purpose |
| ----------------- | --- | ------- |
| `WilorSource`     | ✅  | Camera → YOLO + WiLoR on a worker thread, latest-frame-wins (drops stale frames). Keeps the best detection per side, like `mano_scene.py`. |
| `ReplaySource`    | ✅  | Plays an `encord-scene/out/<episode>/mano.npz` (plus `frames/` for the video pane) in real time. Lets us develop and demo the UI without a camera or GPU, and proves the boundary holds. |
| `SyntheticSource` | ✅  | Generates joint angles (sinusoidal finger curls) → `JointAngleAdapter` → `HandFrame`. A stand-in for the EMG path before hardware exists. |
| `EmgSource`       | later | Wristband driver → joint angles → `JointAngleAdapter`. Possibly wrist orientation from the band IMU (the episodes already carry `/imu` topics). |

`JointAngleAdapter` turns anatomical angles (per joint: flexion, abduction, maybe twist, in degrees or radians)
into MANO `hand_pose` axis-angle, using per-joint local axes taken from the MANO rest pose. It's the only
EMG-specific maths, and we can build and test it now against `SyntheticSource`, then fit it to the real
wristband format once we know what that is.

### Threads

- **Qt main thread:** UI, video blit, 3D redraw, driven by a ~60 Hz `QTimer` that renders the latest state.
- **Capture thread:** `cv2.VideoCapture.read()` in a loop. Publishes the latest frame. Never queues.
- **Pose thread:** one per source. WiLoR runs here (MPS/CUDA). Results go back to the main thread via a Qt
  signal (`QObject` bridge wrapping `emit`).
- The smoother and MANO forward pass run on the main thread (sub-ms).

### Smoothing / latency

WiLoR updates at ~5 fps and jitters frame to frame. Two options, both behind a toggle:

1. **One-Euro filter** on parameters (axis-angle → quaternion for rotations, plain for transl/betas). Cheap,
   standard for hand tracking, tunable lag versus jitter.
2. **Hold-last plus a short fade-out** when a hand disappears (about 300 ms, then hide), instead of popping.

Betas should be frozen per hand after the first N confident frames. Hand shape doesn't change, and letting
it float adds jitter.

### UI

`QMainWindow` with a horizontal `QSplitter`:

- **Left:** `QLabel`/`QGraphicsView` showing the camera frame. Optional overlay of WiLoR boxes and the projected
  mesh outline, which is a cheap visual check of alignment.
- **Right:** 3D view. Two hand meshes (left/right coloured differently), a ground grid, an orbit camera, and a
  "camera view" button that snaps to the video's viewpoint so the render lines up with the left pane.
- **Toolbar:** source picker (Camera+WiLoR / Replay / Synthetic / EMG), camera device, device (mps/cuda/cpu),
  smoothing toggle, FPS readouts (video fps, pose fps, pose latency).

## Technology choices

| Concern      | Recommendation | Why / alternatives |
| ------------ | -------------- | ------------------ |
| Qt binding   | **PySide6**    | Same API as PyQt6 but LGPL; PyQt6 is GPL or commercial. If you specifically want PyQt6 it's a one-line import shim (`qtpy`). |
| 3D           | **pyqtgraph.opengl** (`GLViewWidget` + `GLMeshItem.setMeshData` per update) | Light, pure Python, and 2×778 verts per frame is trivial. Fallback if it looks too flat: **pyvistaqt** (VTK, much nicer shading, heavier). The renderer sits behind a tiny `HandRenderer` interface so this stays swappable. |
| Camera       | OpenCV `VideoCapture` (AVFoundation on macOS) | Already a dependency. macOS asks for camera permission for the terminal or app the first time. |
| Pose model   | WiLoR-mini (as now) | Proven in this repo. If 5 fps feels too laggy: run on CUDA, run WiLoR only every N frames, or use MediaPipe Hands (real-time, 21 keypoints) with MANO IK. The swap is just another `PoseSource`. |
| Python / env | Own `pyproject.toml` + `uv`, Python 3.11 | Still pinned by wilor-mini's `torch==2.5.0`. Copy the `[tool.uv.sources]` overrides (wilor-mini git, chumpy git). |

## Layout

```
hand-viewer/
  pyproject.toml
  hand_viewer/
    __main__.py              # uv run python -m hand_viewer [--source camera|replay|synthetic] ...
    core/types.py            # HandPose, HandFrame, PoseSource, FrameSource — the contract
    core/mano.py             # MANO forward (+ left mirroring/winding), loads MANO_RIGHT.pkl from ~/.cache/wilor-mini
    core/smoothing.py        # One-Euro filter on HandPose
    core/joint_angles.py     # JointAngleAdapter: anatomical angles → MANO hand_pose
    sources/camera.py        # FrameSource: OpenCV capture thread
    sources/wilor.py         # PoseSource: YOLO + WiLoR
    sources/replay.py        # PoseSource + FrameSource from encord-scene/out/<ep>/
    sources/synthetic.py     # PoseSource via JointAngleAdapter
    ui/main_window.py
    ui/video_view.py
    ui/hand_view.py          # HandRenderer (pyqtgraph)
  tests/                     # contract + MANO + adapter tests; no Qt or GPU needed
```

## Build order

1. Scaffold, types, `core/mano.py`. Test: MANO of zeros = flat hand; left mirror matches WiLoR's left vertices
   in `mano.npz`.
2. Window with `ReplaySource`: video left, meshes right, in sync. This proves the UI and the boundary with no
   camera or GPU.
3. `SyntheticSource` + `JointAngleAdapter`: fingers curl as expected, and `transl=None` placement works.
4. `FrameSource` camera + `WilorSource` on a worker thread: live demo. Measure end-to-end latency.
5. Smoothing, overlay, camera-view snap, FPS readouts.
6. (Later) `EmgSource` once the wristband output format is known.

## Open questions

1. **EMG output format.** Which joints, which DoF per joint (flex/abd/twist), units, rate. Does it include wrist
   orientation (band IMU) or position? This decides `JointAngleAdapter`'s input and whether the EMG hand floats
   at a fixed anchor.
2. **Video pane in EMG mode.** Do we still want the camera on the left as ground truth, or replace it with an
   EMG plot?
3. **PySide6 vs PyQt6.** Is licensing a concern, or do you specifically want PyQt6?
4. **Target hardware for live demos.** Mac only (about 5 fps pose), or also a CUDA box?
5. **Webcam intrinsics.** For a generic webcam we'll assume about a 60° horizontal FOV unless we calibrate. That
   only affects absolute depth, not finger pose.

## Review changes (2026-10-06)

Two reviews (architecture/EMG readiness, technical feasibility). Applied:

1. Left-hand convention corrected (see contract). `ManoModel` must reproduce `left_vertices` in `mano.npz`; that
   test gates everything else.
2. `global_orient`/`transl`/`betas` optional, plus `frame` (camera/world). Unposed hands are anchored by their
   root joint. "Zero heading" control for world-frame orientation.
3. Shared monotonic clock stamped at capture. The session keeps a short video ring buffer so it can show the frame
   matching the rendered pose ("sync video to pose").
4. Explicit lifecycle (non-blocking start, joining stop, status/error), and a `Session` that owns the source pair
   with a run id. `core/` stays Qt-free: sinks are lock-protected buffers drained by the UI timer.
5. Smoothing: One-Euro on quaternions (with a hemisphere check) plus render-delay interpolation, config per
   source, betas frozen to the median of the first N frames.
6. WiLoR: our own crop-then-blur preprocessing (WiLoR blurs the whole image per hand, about 52 ms each), fp16 on
   MPS to measure, handedness hysteresis and box matching across frames.
7. Camera via Qt Multimedia (`QCamera` + `QVideoSink`, `QCameraPermission`) instead of OpenCV capture.
8. `SyntheticSource` writes MANO `hand_pose` directly. `JointAngleAdapter` is deferred until the band's format is
   known (emg2pose uses a 20-DoF UmeTrack skeleton, so this is retargeting).
9. Deferred: a generic HandFrame recorder/log format (Replay reads `mano.npz` for now), and an
   `opencv-python-headless` swap for the Linux box.
