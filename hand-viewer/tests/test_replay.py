"""Replay: shared clock, lifecycle, poses and frame decoding. Mostly on a tiny synthetic episode (fast, deterministic);
one smoke test on a real encord-scene episode when it's present."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from hand_viewer.core.types import SIDES, FrameSource, PoseSource, now_ns
from hand_viewer.sources.replay import Replay, decode_frame

REAL_EPISODE = Path(__file__).resolve().parents[2] / "encord-scene/out/sub-P001Fer_task-Glue_ep-005"
T0 = 1_790_000_000_000_000_000
GAP = 50_000_000  # 20 fps
N = 5
W, H = 64, 40
K = [50.0, 0, 31.5, 0, 52.0, 19.5, 0, 0, 1]


class Sink:
    def __init__(self):
        self.lock = threading.Lock()
        self.items, self.errors, self.statuses = [], [], []

    def _add(self, x):
        with self.lock:
            self.items.append((now_ns(), x))

    video = pose = _add

    def status(self, m):
        self.statuses.append(m)

    def error(self, m):
        self.errors.append(m)

    def snapshot(self):
        with self.lock:
            return list(self.items)


def make_episode(root: Path, *, frames=True, n=N) -> Path:
    rng = np.random.default_rng(0)
    times = T0 + GAP * np.arange(n, dtype=np.int64)
    res = {"log_time_ns": times, "faces": np.zeros((1, 3), np.int32)}
    for side in SIDES:
        for key, shape in {"global_orient": (n, 3), "hand_pose": (n, 45), "betas": (n, 10), "cam_t": (n, 3),
                           "vertices": (n, 778, 3), "bbox": (n, 4)}.items():
            res[f"{side}_{key}"] = rng.normal(size=shape).astype(np.float32)
    res["left_score"] = np.where(np.arange(n) % 2 == 0, 0.9, 0.0).astype(np.float32)  # left on even rows only
    res["right_score"] = np.full(n, 0.7, np.float32)
    np.savez(root / "mano.npz", **res)
    if frames:
        (root / "frames").mkdir()
        for i, t in enumerate(times):
            img = np.full((H, W, 3), (i * 40, 0, 255), np.uint8)  # BGR: red channel 255, blue encodes index
            cv2.imwrite(str(root / "frames" / f"{t}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        (root / "camera.json").write_text(json.dumps({"width": W, "height": H, "k": K, "d": [0] * 5,
                                                      "distortion_model": "plumb_bob"}))
    return root


@pytest.fixture
def episode(tmp_path):
    return make_episode(tmp_path)


def run_both(replay, seconds):
    fs, ps = Sink(), Sink()
    replay.frame_source.start(fs)
    replay.pose_source.start(ps)
    time.sleep(seconds)
    replay.frame_source.stop()
    replay.pose_source.stop()
    return fs, ps


def test_protocols(episode):
    r = Replay(episode)
    assert isinstance(r.frame_source, FrameSource) and isinstance(r.pose_source, PoseSource)
    assert r.frame_source.name == r.pose_source.name == "replay"


def test_shared_clock_same_image_same_t_ns(episode):
    fs, ps = run_both(Replay(episode, speed=2.0), 0.4)
    assert not fs.errors and not ps.errors
    video = {(f.meta["loop"], f.meta["index"]): f.t_ns for _, f in fs.snapshot()}
    poses = {f.t_ns for _, f in ps.snapshot()}
    assert len(video) >= N
    assert set(video.values()) <= poses | {max(video.values())}  # the very last frame may beat its pose to the stop


def test_late_starter_joins_the_running_clock(episode):
    r = Replay(episode, speed=1.0)
    ps, fs = Sink(), Sink()
    r.pose_source.start(ps)
    time.sleep(0.12)  # ~2.4 rows in
    r.frame_source.start(fs)
    time.sleep(0.1)
    r.frame_source.stop(), r.pose_source.stop()
    first = fs.snapshot()[0][1]
    assert first.meta["index"] >= 2  # skipped what was already past, rather than starting from 0
    assert first.t_ns in {f.t_ns for _, f in ps.snapshot()}


def test_monotonic_across_loops(episode):
    speed = 4.0
    fs, ps = run_both(Replay(episode, speed=speed), 0.35)  # one loop is 5 * 50 ms / 4 = 62.5 ms
    for sink in (fs, ps):
        t = np.array([f.t_ns for _, f in sink.snapshot()])
        assert len(t) > 2 * N
        assert np.all(np.diff(t) > 0)
        # with no drops every step (including the wrap) is one frame gap of log time
        assert np.all(np.abs(np.diff(t) - GAP / speed) <= GAP / speed + 1)
    loops = [f.meta["loop"] for _, f in fs.snapshot()]
    assert max(loops) >= 2 and loops == sorted(loops)


def test_speed_scaling_and_never_early(episode):
    for speed in (0.5, 2.0):
        r = Replay(episode, speed=speed, loop=False)
        ps = Sink()
        r.pose_source.start(ps)
        time.sleep(GAP * N / speed / 1e9 + 0.1)
        r.pose_source.stop()
        items = ps.snapshot()
        t = np.array([f.t_ns for _, f in items])
        assert len(t) == N
        np.testing.assert_allclose(np.diff(t), GAP / speed, atol=1)
        assert all(recv >= f.t_ns for recv, f in items)  # emitted at (not before) its due time
        assert ps.statuses[-1] == "Replay finished"


def test_stop_semantics(episode):
    r = Replay(episode, speed=1.0)
    r.pose_source.stop()  # stop before start is fine
    fs, ps = Sink(), Sink()
    r.frame_source.start(fs)
    r.pose_source.start(ps)
    time.sleep(0.08)
    r.frame_source.stop()
    n_video = len(fs.snapshot())
    time.sleep(0.15)
    assert len(fs.snapshot()) == n_video  # nothing after stop returned
    assert len(ps.snapshot()) > 2  # stopping one leaves the other running
    r.pose_source.stop()
    r.pose_source.stop()  # idempotent
    n_pose = len(ps.snapshot())
    time.sleep(0.1)
    assert len(ps.snapshot()) == n_pose
    with pytest.raises(RuntimeError):
        r.pose_source.start(Sink())


def test_stop_interrupts_long_sleep(tmp_path):
    r = Replay(make_episode(tmp_path), speed=0.001)  # 50 s between rows
    ps = Sink()
    r.pose_source.start(ps)
    time.sleep(0.02)
    start = time.monotonic()
    r.pose_source.stop()
    assert time.monotonic() - start < 0.5
    assert len(ps.snapshot()) == 1


def test_poses_for_known_row(episode):
    r = Replay(episode)
    m = np.load(episode / "mano.npz")
    f1 = r.pose_source.hand_frame(1, 123)
    assert f1.t_ns == 123 and f1.source == "replay" and set(f1.hands) == {"right"}  # left score 0 on odd rows
    f2 = r.pose_source.hand_frame(2, 0)
    assert set(f2.hands) == {"left", "right"}
    for side, score in (("left", 0.9), ("right", 0.7)):
        h = f2.hands[side]
        np.testing.assert_array_equal(h.hand_pose, m[f"{side}_hand_pose"][2].reshape(15, 3))
        np.testing.assert_array_equal(h.global_orient, m[f"{side}_global_orient"][2])  # left passed through as-is
        np.testing.assert_array_equal(h.transl, m[f"{side}_cam_t"][2])
        np.testing.assert_array_equal(h.betas, m[f"{side}_betas"][2])
        assert h.frame == "camera" and h.confidence == pytest.approx(score)


def test_frame_content_and_intrinsics(episode):
    fs, _ = run_both(Replay(episode, speed=4.0, max_long_side=None), 0.1)
    f = fs.snapshot()[0][1]
    assert f.image.shape == (H, W, 3) and f.image.dtype == np.uint8 and f.image.flags.c_contiguous
    assert f.image[..., 0].mean() > 250 and f.image[..., 2].mean() == pytest.approx(f.meta["index"] * 40, abs=3)  # RGB
    assert f.meta["log_time_ns"] == T0 + GAP * f.meta["index"]
    i = f.intrinsics
    assert (i.width, i.height, i.fx, i.fy, i.cx, i.cy) == (W, H, K[0], K[4], K[2], K[5])


def test_downscale_scales_intrinsics(episode):
    path = sorted((episode / "frames").glob("*.jpg"))[0]
    camera = json.loads((episode / "camera.json").read_text())
    img, i = decode_frame(path, camera, 32)
    assert img.shape == (20, 32, 3) and img.flags.c_contiguous
    assert (i.width, i.height) == (32, 20)
    assert (i.fx, i.fy) == pytest.approx((K[0] / 2, K[4] / 2))
    assert (i.cx, i.cy) == pytest.approx((15.5, 9.5))  # image centre stays the image centre
    assert decode_frame(path, camera, 1000)[0].shape == (H, W, 3)  # never upscales


def test_pose_only_without_frames(tmp_path):
    r = Replay(make_episode(tmp_path, frames=False), speed=4.0)
    fs, ps = run_both(r, 0.1)
    assert len(ps.snapshot()) > 0 and not ps.errors
    assert fs.errors and "frames" in fs.errors[0] and not fs.snapshot()


def test_validation(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Replay(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="mano.npz"):
        Replay(tmp_path)
    ep = make_episode(tmp_path)
    with pytest.raises(ValueError, match="speed"):
        Replay(ep, speed=0)
    (ep / "camera.json").unlink()
    with pytest.raises(FileNotFoundError, match="camera.json"):
        Replay(ep)
    m = dict(np.load(ep / "mano.npz"))
    m["right_hand_pose"] = m["right_hand_pose"][:, :44]
    np.savez(ep / "mano.npz", **m)
    with pytest.raises(ValueError, match="right_hand_pose"):
        Replay(ep)


@pytest.mark.skipif(not REAL_EPISODE.is_dir(), reason="real episode not downloaded")
def test_real_episode_smoke():
    r = Replay(REAL_EPISODE)
    assert r.num_frames == r.num_rows > 0
    fs, ps = run_both(r, 0.5)
    assert not fs.errors and not ps.errors
    frames, poses = [f for _, f in fs.snapshot()], [f for _, f in ps.snapshot()]
    assert len(frames) >= 4 and len(poses) >= 4
    f = frames[0]
    assert f.image.shape == (600, 960, 3) and f.intrinsics.width == 960
    assert f.intrinsics.fx == pytest.approx(734.2565796842308 / 2)
    assert {p.t_ns for p in poses} >= {f.t_ns for f in frames[:-1]}
