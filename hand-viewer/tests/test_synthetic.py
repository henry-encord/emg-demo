import threading
import time
from pathlib import Path

import numpy as np
import pytest

from hand_viewer.core.types import PoseSource
from hand_viewer.sources.synthetic import FINGERS, MIRROR, SyntheticSource, curl_pose

MANO_PATH = Path("~/.cache/wilor-mini/pretrained_models/MANO_RIGHT.pkl").expanduser()
TIP_VERTS = {"index": 320, "middle": 443, "pinky": 672, "ring": 555, "thumb": 744}  # standard MANO fingertips
BASE_JOINTS = {"index": 1, "middle": 4, "pinky": 7, "ring": 10}  # MCP joints (MANO joint order)


class Sink:
    def __init__(self):
        self.frames, self.errors, self.statuses = [], [], []
        self.lock = threading.Lock()

    def pose(self, frame):
        with self.lock:
            self.frames.append(frame)

    def status(self, message):
        self.statuses.append(message)

    def error(self, message):
        self.errors.append(message)


def test_lifecycle_rate_and_no_emits_after_stop():
    src = SyntheticSource(rate_hz=100)
    assert isinstance(src, PoseSource) and src.name == "synthetic"
    sink = Sink()
    t0 = time.monotonic()
    src.start(sink)
    assert time.monotonic() - t0 < 0.05  # non-blocking
    time.sleep(0.5)
    src.stop()
    n = len(sink.frames)
    time.sleep(0.05)
    assert len(sink.frames) == n
    assert 35 <= n <= 60, n
    assert not sink.errors and sink.statuses
    f = sink.frames[-1]
    assert set(f.hands) == {"left", "right"} and f.source == "synthetic"
    h = f.hands["right"]
    assert h.global_orient is None and h.transl is None and h.betas is None
    ts = [x.t_ns for x in sink.frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    src.stop()  # idempotent
    with pytest.raises(RuntimeError):
        src.start(sink)


def test_stop_without_start_and_stop_from_sink():
    SyntheticSource().stop()
    src = SyntheticSource(rate_hz=200)

    class StoppingSink(Sink):
        def pose(self, frame):
            super().pose(frame)
            src.stop()  # from the worker thread: must not deadlock

    sink = StoppingSink()
    src.start(sink)
    time.sleep(0.1)
    src.stop()
    assert len(sink.frames) == 1


def test_sink_error_reported():
    class BadSink(Sink):
        def pose(self, frame):
            raise ValueError("boom")

    sink, src = BadSink(), SyntheticSource()
    src.start(sink)
    time.sleep(0.1)
    src.stop()
    assert sink.errors and "boom" in sink.errors[0]


def test_left_is_mirror_of_right():
    curl = np.linspace(0, 1, 5)
    np.testing.assert_allclose(curl_pose(curl, "left"), curl_pose(curl, "right") * MIRROR)
    f = SyntheticSource().frame_at(1_234_000_000)
    assert f.hands["left"].hand_pose.shape == (15, 3)


@pytest.mark.skipif(not MANO_PATH.exists(), reason="MANO_RIGHT.pkl not downloaded")
def test_curl_closes_fingers():
    torch = pytest.importorskip("torch")
    smplx = pytest.importorskip("smplx")
    mano = smplx.MANO(str(MANO_PATH), use_pca=False, flat_hand_mean=True, is_rhand=True)

    def run(hand_pose):
        out = mano(hand_pose=torch.tensor(hand_pose, dtype=torch.float32).reshape(1, 45),
                   global_orient=torch.zeros(1, 3), betas=torch.zeros(1, 10))
        return out.vertices[0].detach().numpy(), out.joints[0].detach().numpy()

    v_open, joints = run(curl_pose(np.zeros(5)))
    v_fist, _ = run(curl_pose(np.ones(5)))
    palm = joints[[0, 1, 4, 7, 10]].mean(0)  # wrist + finger bases
    for f in FINGERS:
        i = TIP_VERTS[f]
        ref = joints[BASE_JOINTS[f]] if f in BASE_JOINTS else palm  # fingers fold onto their base; thumb into palm
        d_open, d_fist = np.linalg.norm(v_open[i] - ref), np.linalg.norm(v_fist[i] - ref)
        assert d_fist < 0.7 * d_open, (f, d_open, d_fist)
        assert v_fist[i, 1] < v_open[i, 1], f  # towards the palm side (-y in MANO rest pose)
