"""Live WiLoR pose source: YOLO hand detector + WiLoR-mini on a worker thread, fed the latest video frame.

Same model and conventions as encord-scene's `run_wilor`, but with our own patch preprocessing:
`predict_with_bboxes` Gaussian-blurs the *whole* image once per hand (~52 ms/hand at 1920 px, holding the GIL),
whereas the patch only ever samples a box around the hand. We blur just that box (padded by the kernel radius, so
the result is the same) and run all hands through the ViT in one batch.

torch / wilor_mini are imported lazily so importing this module (e.g. to list sources in a menu) stays cheap.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from hand_viewer.core.types import (
    SIDES, CameraIntrinsics, HandFrame, HandPose, PoseSink, Side, VideoFrame,
)

WEIGHTS_DIR = Path.home() / ".cache" / "wilor-mini"
PATCH = 256           # WiLoR's input patch size
RESCALE_FACTOR = 2.5  # patch side = 2.5 x the detector box's longer side (predict_with_bboxes default)


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_fp16(device) -> bool:
    # cuda: fp16 is the standard speed-up. mps: measured on the M4 Pro (scripts/bench_wilor.py, 40 frames, 2 hands)
    # the ViT-H is compute-bound and fp16 gave no consistent gain (pose 127-140 ms vs 131-137 ms fp32 across runs,
    # 0-3%) while moving vertices ~0.3 mm and cam_t ~0.9 mm. Accurate enough, but not worth it: off. cpu: never.
    return device.type == "cuda"


# ---------------------------------------------------------------------------------------------- handedness


@dataclass(frozen=True)
class Detection:
    box: np.ndarray  # (4,) xyxy pixels
    conf: float
    is_right: bool   # the detector's class


@dataclass(frozen=True)
class Track:
    """What we believe about one side from previous frames."""

    box: np.ndarray
    disagree: int = 0  # consecutive frames the detector's class said the other side
    missing: int = 0   # consecutive frames without a matching detection


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _near(track_box: np.ndarray, box: np.ndarray) -> float:
    """Centre distance in units of the track box's diagonal (hands move a lot between ~6 fps pose updates)."""
    diag = np.hypot(track_box[2] - track_box[0], track_box[3] - track_box[1]) + 1e-6
    return float(np.hypot(*((box[:2] + box[2:]) / 2 - (track_box[:2] + track_box[2:]) / 2)) / diag)


def assign_sides(dets: list[Detection], tracks: dict[Side, Track], *, switch_after: int = 3, max_missing: int = 3,
                 min_iou: float = 0.1, max_dist: float = 0.6) -> tuple[dict[Side, Detection], dict[Side, Track]]:
    """At most one detection per side, with handedness hysteresis. Pure: returns (assignment, new tracks).

    YOLO's left/right class flickers frame to frame (and sometimes calls both hands "right"). A detection that
    overlaps last frame's box for a side inherits that side; the detector only gets to relabel it after disagreeing
    `switch_after` frames in a row. Unmatched detections take the detector's class. Tracks survive `max_missing`
    frames without a detection so a brief dropout doesn't reset them.
    """
    pairs = []
    for side, tr in tracks.items():
        for j, d in enumerate(dets):
            iou, dist = _iou(tr.box, d.box), _near(tr.box, d.box)
            if iou >= min_iou or dist <= max_dist:
                pairs.append((-iou, dist, side, j))
    matched: dict[int, Side] = {}
    for _, _, side, j in sorted(pairs, key=lambda p: (p[0], p[1])):
        if side not in matched.values() and j not in matched:
            matched[j] = side

    # candidates: (priority, side, detection, disagree); priority prefers stable matched tracks, then confidence
    cands = []
    for j, d in enumerate(dets):
        said: Side = SIDES[int(d.is_right)]
        if j in matched:
            side, disagree = matched[j], tracks[matched[j]].disagree
            disagree = disagree + 1 if said != side else 0
            if disagree >= switch_after:
                side, disagree = said, 0
            cands.append(((side == matched[j], d.conf), side, d, disagree))
        else:
            cands.append(((False, d.conf), said, d, 0))
    out: dict[Side, Detection] = {}
    new_tracks: dict[Side, Track] = {}
    for _, side, d, disagree in sorted(cands, key=lambda c: c[0], reverse=True):
        if side not in out:
            out[side] = d
            new_tracks[side] = Track(np.asarray(d.box, np.float64), disagree, 0)
    for side, tr in tracks.items():
        if side not in new_tracks and tr.missing + 1 <= max_missing:
            new_tracks[side] = replace(tr, missing=tr.missing + 1)
    return out, new_tracks


# ---------------------------------------------------------------------------------------------- model


def make_patch(image: np.ndarray, center: np.ndarray, size: float, flip: bool) -> np.ndarray:
    """The 256x256 patch `predict_with_bboxes` makes, blurring only the region the patch samples from.

    Blur sigma is WiLoR's ((size / 256 / 2) - 1) / 2 when that downsampling factor > 1.1. cv2.GaussianBlur with
    radius int(4 sigma + 0.5) and BORDER_REPLICATE is skimage's gaussian (truncate=4, mode='nearest'); padding the
    crop by that radius makes the blurred values inside the patch's footprint identical to blurring everything.
    """
    from wilor_mini.utils import utils

    h, w = image.shape[:2]
    factor = size / PATCH / 2
    sigma = (factor - 1) / 2 if factor > 1.1 else 0.0
    radius = int(4 * sigma + 0.5) if sigma else 0
    pad = radius + 2  # +2 for bilinear sampling at the patch edge
    x0 = max(int(np.floor(center[0] - size / 2)) - pad, 0)
    y0 = max(int(np.floor(center[1] - size / 2)) - pad, 0)
    x1 = min(int(np.ceil(center[0] + size / 2)) + pad + 1, w)
    y1 = min(int(np.ceil(center[1] + size / 2)) + pad + 1, h)
    crop = image[y0:y1, x0:x1]
    if sigma:
        crop = cv2.GaussianBlur(crop.astype(np.float32), (2 * radius + 1,) * 2, sigma,
                                borderType=cv2.BORDER_REPLICATE)
    # Outside the image the crop is clipped, so warpAffine's zero border still applies exactly where WiLoR's did.
    patch, _ = utils.generate_image_patch_cv2(crop, center[0] - x0, center[1] - y0, size, size, PATCH, PATCH,
                                              flip, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
    return patch


@dataclass(frozen=True)
class HandResult:
    side: Side
    detection: Detection
    global_orient: np.ndarray  # (3,)
    hand_pose: np.ndarray      # (15, 3)
    betas: np.ndarray          # (10,)
    cam_t: np.ndarray          # (3,) camera frame, principal-point corrected
    vertices: np.ndarray       # (778, 3) WiLoR's (left already mirrored), before cam_t; for checks/overlays


class WilorModel:
    """YOLO + WiLoR without threads; used by WilorSource, the benchmark and the tests.

    Images are BGR: ultralytics treats numpy input as BGR, and WiLoR's forward flips channels BGR -> RGB before its
    ImageNet normalisation (WiLoR-mini's README feeds RGB, contradicting its own code). Measured on the Glue episode,
    BGR raises detector confidence by 0.02-0.05, stops a dropped left hand, and lowers joint jitter ~5-15%."""

    def __init__(self, device="auto", fp16: bool | None = None):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

        self.device = pick_device(device) if isinstance(device, str) else device
        self.fp16 = default_fp16(self.device) if fp16 is None else fp16
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.pipe = WiLorHandPose3dEstimationPipeline(
            device=self.device, dtype=self.dtype, verbose=False, wilor_pretrained_dir=str(WEIGHTS_DIR))
        # WiLoR feeds its ViT a width-sliced (non-contiguous) tensor, which MPS convolutions reject
        # ("view size is not compatible with input tensor's size and stride"); hand every conv a contiguous input.
        for module in self.pipe.wilor_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.register_forward_pre_hook(lambda _, inputs: (inputs[0].contiguous(), *inputs[1:]))

    def set_focal(self, fx: float, width: int, height: int) -> None:
        """WiLoR's full-image focal is focal_length / 256 * max(w, h); choose it so that equals the real fx. The
        RefineNet also uses it (as the crop camera), so it must be set on the model, not just in post-processing."""
        f = fx * PATCH / max(width, height)
        self.pipe.FOCAL_LENGTH = f
        self.pipe.wilor_model.FOCAL_LENGTH = f

    def detect(self, image: np.ndarray, conf: float = 0.3) -> list[Detection]:
        boxes = self.pipe.hand_detector(image, conf=conf, verbose=False, device=str(self.device))[0].boxes
        return [Detection(np.asarray(b, np.float64), float(c), bool(k))
                for b, c, k in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]

    def estimate(self, image: np.ndarray, hands: dict[Side, Detection], k: CameraIntrinsics) -> list[HandResult]:
        """WiLoR on the given boxes (one batch); post-processing replicates predict_with_bboxes."""
        import torch

        if not hands:
            return []
        from wilor_mini.utils import utils

        h, w = image.shape[:2]
        self.set_focal(k.fx, w, h)
        sides = list(hands)
        boxes = np.stack([hands[s].box for s in sides])
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        sizes = (RESCALE_FACTOR * (boxes[:, 2:4] - boxes[:, 0:2])).max(axis=1)
        patches = np.stack([make_patch(image, c, s, side == "left").astype(np.float32)
                            for side, c, s in zip(sides, centers, sizes)])
        with torch.no_grad():
            out = self.pipe.wilor_model(torch.from_numpy(patches).to(device=self.device, dtype=self.dtype))
        out = {key: v.cpu().float().numpy() for key, v in out.items()}

        focal = self.pipe.FOCAL_LENGTH / PATCH * max(w, h)  # == k.fx
        img_size = np.array([[w, h]], np.float64)
        results = []
        for i, side in enumerate(sides):
            pred_cam = out["pred_cam"][i:i + 1].copy()
            go, hp, verts = out["global_orient"][i].reshape(3), out["hand_pose"][i].reshape(15, 3), out[
                "pred_vertices"][i].copy()
            if side == "left":  # WiLoR ran the flipped crop through its right-hand model; un-mirror
                pred_cam[:, 1] *= -1
                go = go * np.array([1, -1, -1], np.float32)
                hp = hp * np.array([1, -1, -1], np.float32)
                verts[:, 0] *= -1
            cam_t = utils.cam_crop_to_full(pred_cam, centers[i][None], sizes[i], img_size, focal)[0]
            # WiLoR assumes the principal point is the image centre; shift to the real one.
            cam_t[0] += (w / 2 - k.cx) * cam_t[2] / k.fx
            cam_t[1] += (h / 2 - k.cy) * cam_t[2] / k.fy
            results.append(HandResult(side, hands[side], go, hp, out["betas"][i].reshape(10), cam_t, verts))
        return results


def prepare(frame: VideoFrame, long_side: int) -> tuple[np.ndarray, CameraIntrinsics]:
    """Downscale so the long side <= long_side, scaling the intrinsics to match, and convert to BGR (WilorModel)."""
    image = frame.image
    h, w = image.shape[:2]
    k = frame.intrinsics or CameraIntrinsics.from_fov(w, h)
    if (k.width, k.height) != (w, h):  # intrinsics given for another resolution: rescale to this image first
        k = _scale_k(k, w, h)
    s = long_side / max(w, h)
    if s < 1:
        nw, nh = round(w * s), round(h * s)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        k = _scale_k(k, nw, nh)
    return np.ascontiguousarray(image[..., ::-1]), k


def _scale_k(k: CameraIntrinsics, w: int, h: int) -> CameraIntrinsics:
    sx, sy = w / k.width, h / k.height
    # pixel centres: x' + 0.5 = (x + 0.5) * s
    return CameraIntrinsics(w, h, k.fx * sx, k.fy * sy, (k.cx + 0.5) * sx - 0.5, (k.cy + 0.5) * sy - 0.5)


# ---------------------------------------------------------------------------------------------- source


@dataclass(frozen=True)
class WilorStats:
    detect_ms: float = 0.0  # last frame: downscale + YOLO
    pose_ms: float = 0.0    # last frame: patches + WiLoR + post-processing
    total_ms: float = 0.0
    fps: float = 0.0        # pose updates/s (completion to completion), exponentially smoothed
    processed: int = 0
    dropped: int = 0       # frames replaced in the slot before the worker got to them


class WilorSource:
    """PoseSource + FrameConsumer. `submit_frame` never blocks: it overwrites a one-frame slot, and the worker
    always takes the newest frame. `stats` is replaced atomically, so any thread can read it."""

    name = "wilor"

    def __init__(self, device: str = "auto", *, input_long_side: int = 960, fp16: bool | None = None,
                 det_conf: float = 0.3, switch_after: int = 3):
        self.device, self.input_long_side, self.fp16, self.det_conf = device, input_long_side, fp16, det_conf
        self.switch_after = switch_after
        self.stats = WilorStats()
        self._cond = threading.Condition()
        self._slot: VideoFrame | None = None
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._tracks: dict[Side, Track] = {}
        self._dropped = 0
        self._last_done: float | None = None

    def start(self, sink: PoseSink) -> None:
        if self._thread is not None:
            raise RuntimeError("WilorSource can only be started once")
        self._thread = threading.Thread(target=self._run, args=(sink,), name="wilor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()  # may wait for an in-flight model load / inference to finish

    def submit_frame(self, frame: VideoFrame) -> None:
        with self._cond:
            if self._slot is not None:
                self._dropped += 1
            self._slot = frame
            self._cond.notify()

    def _next_frame(self) -> VideoFrame | None:
        with self._cond:
            while self._slot is None and not self._stopping:
                self._cond.wait()
            if self._stopping:
                return None
            frame, self._slot = self._slot, None
            return frame

    def _run(self, sink: PoseSink) -> None:
        try:
            sink.status("Loading WiLoR…")
            model = self._load()
        except Exception as e:  # noqa: BLE001 - report anything (missing weights, bad device) to the session
            sink.error(f"WiLoR failed to load: {type(e).__name__}: {e}")
            return
        if self._stopping:
            return
        sink.status(f"WiLoR on {model.device.type} ({'fp16' if model.fp16 else 'fp32'})")
        while (frame := self._next_frame()) is not None:
            try:
                hand_frame = self._process(model, frame)
            except Exception as e:  # noqa: BLE001
                sink.error(f"WiLoR failed: {type(e).__name__}: {e}")
                return
            sink.pose(hand_frame)

    def _load(self) -> WilorModel:
        return WilorModel(self.device, self.fp16)

    def _process(self, model: WilorModel, frame: VideoFrame) -> HandFrame:
        t0 = time.perf_counter()
        image, k = prepare(frame, self.input_long_side)
        dets = model.detect(image, self.det_conf)
        t1 = time.perf_counter()
        hands, self._tracks = assign_sides(dets, self._tracks, switch_after=self.switch_after)
        results = model.estimate(image, hands, k)
        t2 = time.perf_counter()
        prev, last, self._last_done = self.stats, self._last_done, t2
        fps = 0.0 if last is None else 1 / (t2 - last) if prev.fps == 0 else 0.8 * prev.fps + 0.2 / (t2 - last)
        self.stats = WilorStats((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3, fps, prev.processed + 1,
                                self._dropped)
        return HandFrame(
            t_ns=frame.t_ns,  # capture time, not inference time
            hands={r.side: HandPose(r.hand_pose, r.global_orient, r.cam_t, r.betas, "camera", r.detection.conf)
                   for r in results},
            source=self.name,
        )
