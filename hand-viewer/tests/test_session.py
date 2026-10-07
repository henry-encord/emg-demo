"""Session lifecycle, run-id staleness, FrameConsumer forwarding, sync-video selection, stats and errors. No Qt."""

import numpy as np
import pytest

from hand_viewer.app import session as session_mod
from hand_viewer.app.session import Session
from hand_viewer.core.types import HandFrame, HandPose, VideoFrame

MS = 1_000_000


class FakeSmoother:
    """Pass-through: sample() returns the newest pushed frame, so tests don't depend on the real filter."""

    def __init__(self, config):
        self.config = config
        self.frames = []

    def push(self, frame):
        self.frames.append(frame)

    def sample(self, t_ns):
        return self.frames[-1] if self.frames else None

    def reset(self):
        self.frames.clear()


@pytest.fixture(autouse=True)
def fake_smoother(monkeypatch):
    monkeypatch.setattr(session_mod, "PoseSmoother", FakeSmoother)


class FakePose:
    name = "fake-pose"

    def __init__(self, log):
        self.log, self.sink = log, None

    def start(self, sink):
        self.log.append("pose.start")
        self.sink = sink

    def stop(self):
        self.log.append("pose.stop")


class FakeConsumerPose(FakePose):
    def __init__(self, log):
        super().__init__(log)
        self.submitted = []

    def submit_frame(self, frame):
        self.submitted.append(frame)


class FakeFrames:
    name = "fake-frames"

    def __init__(self, log):
        self.log, self.sink = log, None

    def start(self, sink):
        self.log.append("frames.start")
        self.sink = sink

    def stop(self):
        self.log.append("frames.stop")


def hand_frame(t_ns):
    return HandFrame(t_ns=t_ns, hands={"right": HandPose(finger_pose=np.zeros((24, 3)))}, source="fake")


def video_frame(t_ns):
    return VideoFrame(t_ns=t_ns, image=np.zeros((2, 2, 3), np.uint8))


def test_lifecycle_order():
    log = []
    s = Session(FakePose(log), FakeFrames(log))
    s.start()
    assert log == ["pose.start", "frames.start"]
    assert s.running
    s.stop()
    assert log[2:] == ["pose.stop", "frames.stop"]
    assert not s.running
    s.stop()  # idempotent at the session level too
    with pytest.raises(RuntimeError):
        s.start()


def test_stale_callbacks_after_stop_are_dropped():
    log = []
    pose, frames = FakePose(log), FakeFrames(log)
    s = Session(pose, frames)
    s.start()
    pose.sink.pose(hand_frame(1))
    frames.sink.video(video_frame(1))
    s.stop()
    pose.sink.pose(hand_frame(2))
    frames.sink.video(video_frame(2))
    pose.sink.status("late")
    st = s.poll(10)
    assert st.hands.t_ns == 1 and st.video.t_ns == 1
    assert st.stats.pose_frames == 1 and st.stats.video_frames == 1
    assert st.status == []


def test_frames_forwarded_to_frame_consumer():
    log = []
    pose, frames = FakeConsumerPose(log), FakeFrames(log)
    s = Session(pose, frames)
    s.start()
    for t in range(3):
        frames.sink.video(video_frame(t * 33 * MS))
    assert [f.t_ns for f in pose.submitted] == [0, 33 * MS, 66 * MS]
    s.stop()
    frames.sink.video(video_frame(999 * MS))
    assert len(pose.submitted) == 3


def test_sync_video_picks_frame_nearest_rendered_pose():
    log = []
    pose, frames = FakePose(log), FakeFrames(log)
    s = Session(pose, frames)
    s.start()
    for t in range(0, 500, 33):
        frames.sink.video(video_frame(t * MS))
    pose.sink.pose(hand_frame(200 * MS))
    assert s.poll(500 * MS).video.t_ns == 495 * MS          # latest by default
    s.set_sync_video(True)
    assert s.poll(500 * MS).video.t_ns == 198 * MS          # nearest to the pose
    s.stop()


def test_video_ring_buffer_is_bounded_to_about_one_second():
    log = []
    frames = FakeFrames(log)
    s = Session(FakePose(log), frames)
    s.start()
    for t in range(0, 3000, 10):
        frames.sink.video(video_frame(t * MS))
    assert 99 <= len(s._video) <= 102
    s.stop()


def test_stats(monkeypatch):
    clock = [0]
    monkeypatch.setattr(session_mod, "now_ns", lambda: clock[0])
    log = []
    pose, frames = FakePose(log), FakeFrames(log)
    s = Session(pose, frames)
    s.start()
    for i in range(60):
        t = i * 20 * MS                       # video at 50 Hz
        frames.sink.video(video_frame(t))
        if i % 5 == 0:                        # pose at 10 Hz, 80 ms after capture
            clock[0] = t + 80 * MS
            pose.sink.pose(hand_frame(t))
    st = s.poll(60 * 20 * MS).stats
    assert st.video_fps == pytest.approx(50, rel=1e-3)
    assert st.pose_fps == pytest.approx(10, rel=1e-3)
    assert st.pose_latency_ms == pytest.approx(80)
    assert st.video_frames == 60 and st.pose_frames == 12
    assert s.poll(10_000 * MS).stats.video_fps == 0      # stale rates decay to 0
    s.stop()


def test_error_stops_session_and_surfaces_message():
    log = []
    pose, frames = FakePose(log), FakeFrames(log)
    s = Session(pose, frames)
    s.start()
    pose.sink.status("loading")
    pose.sink.error("model exploded")
    assert not s.running
    pose.sink.pose(hand_frame(5))                          # dropped: the run is over
    assert "pose.stop" not in log                          # not joined from the worker's thread
    st = s.poll(10)
    assert st.errors == ["model exploded"] and st.status == ["loading"] and not st.running
    assert st.hands is None
    assert log[-2:] == ["pose.stop", "frames.stop"]        # poll() did the stop on the caller's thread
    assert s.poll(20).errors == []


def test_start_exception_becomes_error():
    class Boom(FakePose):
        def start(self, sink):
            raise OSError("no device")

    log = []
    s = Session(Boom(log), FakeFrames(log))
    s.start()
    st = s.poll(1)
    assert not st.running and "no device" in st.errors[0]
    assert "frames.start" not in log


def test_set_smoothing_replays_recent_poses():
    log = []
    pose = FakePose(log)
    s = Session(pose)
    s.start()
    pose.sink.pose(hand_frame(1))
    s.set_smoothing(session_mod.SmoothingConfig(enabled=False))
    assert s.poll(2).hands.t_ns == 1
    s.stop()


def test_smoother_is_fed_only_from_poll():
    """PoseSmoother isn't thread-safe: pose() (worker threads) only queues; poll() (UI thread) pushes."""
    import threading

    log = []
    pose = FakePose(log)
    s = Session(pose)
    s.start()
    t = threading.Thread(target=lambda: [pose.sink.pose(hand_frame(i)) for i in (1, 2, 3)])
    t.start()
    t.join()
    assert s._smoother.frames == []
    assert s.poll(10).hands.t_ns == 3
    assert [f.t_ns for f in s._smoother.frames] == [1, 2, 3]
    s.stop()
