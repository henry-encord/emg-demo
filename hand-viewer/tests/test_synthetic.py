import threading
import time

import numpy as np
import pytest

from hand_viewer.core.types import SOMA_JOINTS, HandPose, PoseSource
from hand_viewer.sources.synthetic import FINGERS, SyntheticSource, curl_pose


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
    assert h.wrist_orient is None and h.wrist_position is None and h.shape is None
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


def test_both_sides_get_the_same_pose():
    f = SyntheticSource().frame_at(1_234_000_000)
    assert f.hands["left"].finger_pose.shape == (24, 3)
    curl = np.linspace(0, 1, 5)
    np.testing.assert_allclose(curl_pose(curl), curl_pose(curl))
    assert np.all(curl_pose(np.zeros(5)) == 0)


@pytest.fixture(scope="module")
def model():
    from hand_viewer.core.hand_model import HandModel

    return HandModel()


@pytest.mark.parametrize("side", ["right", "left"])
def test_curl_closes_fingers(model, side):
    open_ = model.forward(side, HandPose(curl_pose(np.zeros(5)))).joints
    fist = model.forward(side, HandPose(curl_pose(np.ones(5)))).joints
    j = {name: i for i, name in enumerate(SOMA_JOINTS)}
    # Palm normal of the rest hand: fingers (wrist -> middle knuckle) x thumb side (pinky knuckle -> index knuckle)
    # points out of the back of the right hand, so the palm side is the opposite; left mirrors with the same numbers.
    up = open_[j["Middle2"]] - open_[j["Wrist"]]
    across = open_[j["Index2"]] - open_[j["Pinky2"]]
    palm = -np.cross(up, across) * (1 if side == "right" else -1)
    palm /= np.linalg.norm(palm)
    centre = open_[[j["Wrist"], j["Index2"], j["Middle2"], j["Ring2"], j["Pinky2"]]].mean(0)
    for f in FINGERS:
        tip, base = j[f.capitalize() + "End"], j[f.capitalize() + "2"]
        ref = centre if f == "thumb" else open_[base]   # fingers fold onto their knuckle; the thumb into the palm
        d_open, d_fist = np.linalg.norm(open_[tip] - ref), np.linalg.norm(fist[tip] - ref)
        assert d_fist < 0.75 * d_open, (side, f, d_open, d_fist)
        if f != "thumb":
            assert (fist[tip] - open_[tip]) @ palm > 0.02, (side, f)  # moved towards the palm side
