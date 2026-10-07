"""WilorSource: handedness hysteresis (pure), latest-wins + lifecycle (fake model), equivalence with
predict_with_bboxes (real weights, CPU, skipped if weights or recorded frames are absent)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "yolo"))

from hand_viewer.core.types import CameraIntrinsics, HandFrame, PoseSource, FrameConsumer, VideoFrame  # noqa: E402
from hand_viewer.sources import wilor  # noqa: E402
from hand_viewer.sources.wilor import Detection, Track, WilorSource, assign_sides, prepare  # noqa: E402

EPISODE = Path(__file__).resolve().parents[2] / "encord-scene/out/sub-P001Fer_task-Glue_ep-005"
HAVE_WEIGHTS = all((wilor.WEIGHTS_DIR / "pretrained_models" / f).exists()
                   for f in ("wilor_final.ckpt", "detector.pt", "MANO_RIGHT.pkl", "mano_mean_params.npz"))
HAVE_FRAMES = (EPISODE / "frames").is_dir()
needs_model = pytest.mark.skipif(not (HAVE_WEIGHTS and HAVE_FRAMES), reason="WiLoR weights or episode frames absent")


def det(x, is_right, conf=0.9, size=100):
    return Detection(np.array([x, 100, x + size, 100 + size], float), conf, is_right)


# ---------------------------------------------------------------------------------------------- hysteresis


def test_new_detections_take_detector_class_one_per_side():
    out, tracks = assign_sides([det(0, False, 0.5), det(500, True, 0.9), det(900, True, 0.4)], {})
    assert set(out) == {"left", "right"}
    assert out["right"].conf == 0.9 and out["left"].conf == 0.5  # the weaker "right" is dropped
    assert set(tracks) == {"left", "right"}


def test_class_flicker_needs_k_frames_to_switch():
    tracks = {"right": Track(det(500, True).box)}
    for i in range(2):  # detector says "left" twice: still right
        out, tracks = assign_sides([det(505 + i, False)], tracks, switch_after=3)
        assert set(out) == {"right"} and tracks["right"].disagree == i + 1
    out, tracks = assign_sides([det(510, True)], tracks, switch_after=3)  # agreement resets the count
    assert tracks["right"].disagree == 0
    for _ in range(2):
        out, tracks = assign_sides([det(510, False)], tracks, switch_after=3)
    out, tracks = assign_sides([det(510, False)], tracks, switch_after=3)  # third consecutive: switch
    assert set(out) == {"left"} and set(tracks) == {"left", "right"} and tracks["right"].missing == 1


def test_both_called_right_keep_their_tracks():
    tracks = {"left": Track(det(0, False).box), "right": Track(det(500, True).box)}
    out, _ = assign_sides([det(10, True, 0.95), det(490, True, 0.8)], tracks)
    assert out["left"].box[0] == 10 and out["right"].box[0] == 490


def test_moving_hand_matched_by_centre_distance_and_tracks_expire():
    tracks = {"right": Track(det(500, True).box)}
    out, tracks = assign_sides([det(583, False)], tracks)  # IoU 0.09 but centre within 0.6 diagonals: still right
    assert set(out) == {"right"}
    for i in range(3):
        out, tracks = assign_sides([], tracks, max_missing=3)
        assert out == {} and tracks["right"].missing == i + 1
    _, tracks = assign_sides([], tracks, max_missing=3)
    assert tracks == {}


def test_prepare_scales_intrinsics():
    k = CameraIntrinsics(1920, 1200, 734.0, 735.0, 956.5, 628.3)
    image, k2 = prepare(VideoFrame(0, np.zeros((1200, 1920, 3), np.uint8), k), 960)
    assert image.shape == (600, 960, 3) and (k2.width, k2.height) == (960, 600)
    assert k2.fx == pytest.approx(367.0) and k2.cx == pytest.approx((956.5 + 0.5) / 2 - 0.5)
    image, k3 = prepare(VideoFrame(0, np.zeros((480, 640, 3), np.uint8)), 960)  # no intrinsics, no upscaling
    assert image.shape == (480, 640, 3) and k3 == CameraIntrinsics.from_fov(640, 480)


# ---------------------------------------------------------------------------------------------- lifecycle


class RecordingSink:
    def __init__(self):
        self.poses: list[HandFrame] = []
        self.statuses: list[str] = []
        self.errors: list[str] = []
        self.got = threading.Semaphore(0)

    def pose(self, frame):
        self.poses.append(frame)
        self.got.release()

    def status(self, message):
        self.statuses.append(message)

    def error(self, message):
        self.errors.append(message)
        self.got.release()


class FakeModel:
    """Blocks in detect() until released, so the test controls when the worker is busy."""

    device = type("D", (), {"type": "cpu"})()
    fp16 = False

    def __init__(self):
        self.entered = threading.Semaphore(0)
        self.release = threading.Semaphore(0)

    def detect(self, image, conf):
        self.entered.release()
        assert self.release.acquire(timeout=5)
        return []

    def estimate(self, image, hands, k):
        return []


def fake_source(model) -> WilorSource:
    src = WilorSource("cpu")
    src._load = lambda: model
    src._load_converter = lambda: None  # FakeModel finds no hands, so nothing is ever converted
    return src


def frame(t):
    return VideoFrame(t, np.zeros((48, 64, 3), np.uint8))


def test_latest_frame_wins():
    model, sink = FakeModel(), RecordingSink()
    src = fake_source(model)
    assert isinstance(src, PoseSource) and isinstance(src, FrameConsumer)
    src.start(sink)
    src.submit_frame(frame(1))
    assert model.entered.acquire(timeout=5)  # worker is busy with frame 1
    for t in (2, 3, 4):
        src.submit_frame(frame(t))
    model.release.release()
    assert sink.got.acquire(timeout=5)
    assert model.entered.acquire(timeout=5)  # next frame taken: must be the newest
    model.release.release()
    assert sink.got.acquire(timeout=5)
    src.stop()
    assert [p.t_ns for p in sink.poses] == [1, 4]
    assert all(p.hands == {} and p.source == "wilor" for p in sink.poses)  # empty frames are still emitted
    assert src.stats.processed == 2 and src.stats.dropped == 2
    assert sink.statuses[0].startswith("Loading") and "cpu" in sink.statuses[-1] and not sink.errors


class OneHandModel(FakeModel):
    """FakeModel that always "finds" a right hand, so every frame goes to the converter."""

    def detect(self, image, conf):
        super().detect(image, conf)
        return [wilor.Detection(np.array([0.0, 0, 10, 10]), 0.9, True)]

    def estimate(self, image, hands, k):
        return [SimpleNamespace(side=side, mano=None, detection=d) for side, d in hands.items()]


class BlockingConverter:
    def __init__(self):
        self.entered = threading.Semaphore(0)
        self.release = threading.Semaphore(0)

    def convert_many(self, sides, params, confidence):
        self.entered.release()
        assert self.release.acquire(timeout=5)
        return [SimpleNamespace(side=s) for s in sides]


def test_conversion_overlaps_next_inference():
    model, converter, sink = OneHandModel(), BlockingConverter(), RecordingSink()
    src = fake_source(model)
    src._load_converter = lambda: converter
    src.start(sink)
    src.submit_frame(frame(1))
    assert model.entered.acquire(timeout=5)
    model.release.release()
    assert converter.entered.acquire(timeout=5)  # frame 1 is converting...
    src.submit_frame(frame(2))
    assert model.entered.acquire(timeout=5)      # ...while WiLoR already runs frame 2
    model.release.release()
    converter.release.release()
    assert converter.entered.acquire(timeout=5)
    converter.release.release()
    assert sink.got.acquire(timeout=5) and sink.got.acquire(timeout=5)
    src.stop()
    assert [p.t_ns for p in sink.poses] == [1, 2] and set(sink.poses[0].hands) == {"right"}
    assert src.stats.processed == 2 and not sink.errors


def test_conversion_failure_goes_to_sink_error_and_stops():
    model, sink = OneHandModel(), RecordingSink()
    src = fake_source(model)

    class Boom:
        def convert_many(self, *args):
            raise ValueError("bad fit")

    src._load_converter = Boom
    src.start(sink)
    src.submit_frame(frame(1))
    assert model.entered.acquire(timeout=5)
    model.release.release()
    assert sink.got.acquire(timeout=5)
    src._thread.join(timeout=5)
    assert not src._thread.is_alive() and not src._convert_thread.is_alive()
    assert sink.errors == ["SOMA conversion failed: ValueError: bad fit"] and not sink.poses


def test_stop_is_idempotent_and_silences_sink():
    src, sink = fake_source(FakeModel()), RecordingSink()
    src.stop()  # before start: no-op
    src.start(sink)
    with pytest.raises(RuntimeError):
        src.start(sink)
    src.stop()
    src.stop()
    n = len(sink.statuses) + len(sink.poses) + len(sink.errors)
    src.submit_frame(frame(1))
    assert not src._thread.is_alive() and len(sink.statuses) + len(sink.poses) + len(sink.errors) == n


def test_load_failure_goes_to_sink_error():
    src, sink = WilorSource("cpu"), RecordingSink()

    def boom():
        raise FileNotFoundError("no weights")

    src._load = boom
    src.start(sink)
    assert sink.got.acquire(timeout=5)
    src.stop()
    assert sink.errors and "no weights" in sink.errors[0] and not sink.poses


# ---------------------------------------------------------------------------------------------- real model


@pytest.fixture(scope="module")
def model():
    return wilor.WilorModel("cpu", fp16=False)


@pytest.fixture(scope="module")
def frames():
    import cv2

    cam = json.loads((EPISODE / "camera.json").read_text())
    k = CameraIntrinsics(cam["width"], cam["height"], cam["k"][0], cam["k"][4], cam["k"][2], cam["k"][5])
    paths = sorted((EPISODE / "frames").glob("*.jpg"), key=lambda p: int(p.stem))
    return [VideoFrame(int(p.stem), cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), k) for p in paths[100:141:40]]


@needs_model
def test_matches_predict_with_bboxes(model, frames):
    # At 1920 px the hand crops exceed 563 px, so this exercises the crop-then-blur path.
    for f in frames:
        image, k = prepare(f, 1920)
        hands, _ = assign_sides(model.detect(image), {})
        assert set(hands) == {"left", "right"}
        ours = model.estimate(image, hands, k)
        sides = list(hands)
        refs = model.pipe.predict_with_bboxes(image, np.stack([hands[s].box for s in sides]),
                                              [int(s == "right") for s in sides])
        for r, ref in zip(ours, refs):
            p = ref["wilor_preds"]
            cam_t = p["pred_cam_t_full"][0].copy()
            cam_t[0] += (k.width / 2 - k.cx) * cam_t[2] / k.fx
            cam_t[1] += (k.height / 2 - k.cy) * cam_t[2] / k.fy
            np.testing.assert_allclose(r.vertices, p["pred_vertices"][0], atol=1e-4)  # 0.1 mm
            np.testing.assert_allclose(r.hand_pose, p["hand_pose"][0], atol=1e-3)
            np.testing.assert_allclose(r.global_orient, p["global_orient"].reshape(3), atol=1e-3)
            np.testing.assert_allclose(r.betas, p["betas"][0], atol=1e-3)
            np.testing.assert_allclose(r.cam_t, cam_t, atol=1e-3)


@needs_model
def test_source_end_to_end(model, frames):
    src, sink = WilorSource("cpu", input_long_side=960), RecordingSink()
    src._load = lambda: model
    src.start(sink)
    src.submit_frame(frames[0])
    assert sink.got.acquire(timeout=60)
    src.stop()
    assert not sink.errors
    (hf,) = sink.poses
    assert hf.t_ns == frames[0].t_ns and set(hf.hands) == {"left", "right"}
    for pose in hf.hands.values():
        assert pose.frame == "camera" and 0.3 <= pose.confidence <= 1
        assert pose.finger_pose.shape == (24, 3) and pose.wrist_orient.shape == (3,) and pose.shape is None
        assert 0.1 < pose.wrist_position[2] < 2.0  # metres in front of the camera
    assert src.stats.total_ms > 0 and 0 < src.stats.convert_ms < src.stats.total_ms


def test_converter_load_failure_goes_to_sink_error():
    src, sink = fake_source(FakeModel()), RecordingSink()

    def boom():
        raise FileNotFoundError("no SOMA assets")

    src._load_converter = boom
    src.start(sink)
    assert sink.got.acquire(timeout=5)
    src.stop()
    assert sink.errors and "SOMA" in sink.errors[0] and "no SOMA assets" in sink.errors[0] and not sink.poses
