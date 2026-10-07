import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from hand_viewer.core.smoothing import (PoseSmoother, SmoothingConfig, align_hemisphere, quat_from_rotvec,
                                        rotvec_from_quat, slerp)
from hand_viewer.core.types import HandFrame, HandPose

MS = 1_000_000


def frame(t_ms, finger_pose=0.0, orient=None, transl=None, shape=None, side="right", source="test"):
    hp = np.full((24, 3), finger_pose) if np.isscalar(finger_pose) else finger_pose
    return HandFrame(int(t_ms * MS), {side: HandPose(hp, orient, transl, shape)}, source)


def smoother(**kw):
    kw.setdefault("render_delay_ms", 0)
    return PoseSmoother(SmoothingConfig(**kw))


def test_quat_roundtrip_and_hemisphere():
    v = R.random(50, random_state=0).as_rotvec()
    q = quat_from_rotvec(v)
    np.testing.assert_allclose(R.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec(), v, atol=1e-9)
    np.testing.assert_allclose(rotvec_from_quat(-q), v, atol=1e-9)  # -q is the same rotation
    assert np.all(np.sum(align_hemisphere(-q, q) * q, axis=1) > 0)


def test_slerp_takes_short_way_across_hemispheres():
    a = quat_from_rotvec(np.array([0, 0, 0.1]))
    b = -quat_from_rotvec(np.array([0, 0, 0.3]))  # flipped sign must not send us the long way round
    np.testing.assert_allclose(rotvec_from_quat(slerp(a, b, 0.5)), [0, 0, 0.2], atol=1e-9)


def test_nothing_pushed_returns_none():
    assert smoother().sample(0) is None


def test_step_converges_without_excessive_lag():
    s = smoother(min_cutoff=1.0, beta=0.3)
    s.push(frame(0, 0.0))
    for k in range(1, 6):  # 5 Hz, like WiLoR
        s.push(frame(200 * k, 0.5))
        got = s.sample(200 * k * MS).hands["right"].finger_pose[0, 0]
        if k == 1:
            assert 0.25 < got < 0.5  # moved most of the way after one sample
    assert got == pytest.approx(0.5, abs=0.02)


def test_filter_reduces_jitter():
    rng = np.random.default_rng(0)
    s = smoother(min_cutoff=1.0, beta=0.0)
    out = []
    for k in range(200):
        s.push(frame(16 * k, 0.3 + rng.normal(0, 0.05)))
        out.append(s.sample(16 * k * MS).hands["right"].finger_pose[0, 0])
    assert np.std(out[50:]) < 0.02


def test_hemisphere_flip_in_input_does_not_disturb_filter():
    # Same rotation fed as a rotvec near pi from either side (quaternion sign flips between them).
    s = smoother(min_cutoff=1.0, beta=0.0)
    axis = np.array([0.0, 0.0, 1.0])
    for k in range(20):
        ang = np.pi - 0.01 if k % 2 else -(np.pi - 0.01)  # +179.4 deg and -179.4 deg: 1.1 deg apart
        s.push(frame(16 * k, orient=axis * ang))
    o = s.sample(19 * 16 * MS).hands["right"].wrist_orient
    assert abs(np.linalg.norm(o)) > np.pi - 0.02  # stayed near 180 deg instead of averaging to 0


def test_interpolates_between_samples_with_slerp():
    s = smoother(enabled=True, min_cutoff=1e6)  # huge cutoff: filter is a no-op, isolate interpolation
    s.push(frame(0, orient=[0, 0, 0.0], transl=[0, 0, 0]))
    s.push(frame(100, orient=[0, 0, 1.0], transl=[1, 0, 0]))
    h = s.sample(25 * MS).hands["right"]
    np.testing.assert_allclose(h.wrist_orient, [0, 0, 0.25], atol=1e-5)
    np.testing.assert_allclose(h.wrist_position, [0.25, 0, 0], atol=1e-5)


def test_render_delay_and_hold_newest():
    s = smoother(min_cutoff=1e6, render_delay_ms=50)
    s.push(frame(0, 0.0))
    s.push(frame(100, 1.0))
    f = s.sample(100 * MS)
    assert f.t_ns == 50 * MS and f.source == "test"
    assert f.hands["right"].finger_pose[0, 0] == pytest.approx(0.5, abs=1e-5)
    assert s.sample(200 * MS).hands["right"].finger_pose[0, 0] == pytest.approx(1.0, abs=1e-5)  # past newest: hold


def test_auto_delay_tracks_interval():
    s = PoseSmoother(SmoothingConfig(render_delay_ms=None))
    for k in range(50):
        s.push(frame(200 * k))  # t base 0 isn't on the monotonic clock, so no arrival latency is measured
    assert s.render_delay_ns == pytest.approx(200 * MS, rel=0.01)
    s2 = PoseSmoother(SmoothingConfig(render_delay_ms=None))
    for k in range(50):
        s2.push(frame(900 * k))
    assert s2.render_delay_ns == 400 * MS  # clamped


def test_hold_then_drop_and_late_appearance():
    s = smoother(hold_ms=300)
    s.push(frame(0))
    assert "right" in s.sample(250 * MS).hands
    assert "right" not in s.sample(400 * MS).hands
    s.push(HandFrame(500 * MS, {}, "test"))
    assert s.sample(500 * MS).hands == {}
    d = smoother(render_delay_ms=100)
    d.push(frame(1000))
    assert d.sample(1050 * MS).hands == {}  # at render time 950 the hand hadn't appeared yet


def test_stale_frames_dropped():
    s = smoother(min_cutoff=1e6)
    s.push(frame(100, 1.0))
    s.push(frame(50, 0.0))
    assert s.sample(100 * MS).hands["right"].finger_pose[0, 0] == pytest.approx(1.0)


def test_shape_frozen_to_median_and_reset_after_absence():
    s = smoother(freeze_shape_after=3)
    for k, b in enumerate([1.0, 3.0, 2.0, 100.0, 100.0]):
        s.push(frame(10 * k, shape=np.full(20, b)))
    np.testing.assert_allclose(s.sample(40 * MS).hands["right"].shape, 2.0)
    s.push(frame(3000, shape=np.full(20, 7.0)))  # absent > 2 s: re-estimated
    np.testing.assert_allclose(s.sample(3000 * MS).hands["right"].shape, 7.0)


def test_none_fields_stay_none():
    s = smoother()
    s.push(frame(0, 0.1))
    s.push(frame(16, 0.2))
    h = s.sample(8 * MS).hands["right"]
    assert h.wrist_orient is None and h.wrist_position is None and h.shape is None


def test_orientation_appearing_midstream():
    s = smoother(min_cutoff=1e6)
    s.push(frame(0, orient=[0, 0, 1.0]))
    s.push(frame(16))
    s.push(frame(32, orient=[0, 0, 0.5]))
    assert s.sample(16 * MS).hands["right"].wrist_orient is None
    np.testing.assert_allclose(s.sample(32 * MS).hands["right"].wrist_orient, [0, 0, 0.5], atol=1e-5)


def test_disabled_is_passthrough():
    s = smoother(enabled=False, render_delay_ms=200)
    s.push(frame(0, 0.0))
    f_in = frame(100, 1.0, transl=[1, 2, 3])
    s.push(f_in)
    out = s.sample(100 * MS)
    assert out.t_ns == f_in.t_ns
    assert out.hands["right"] is f_in.hands["right"]


def test_two_hands_independent_and_source_change_resets():
    s = smoother(min_cutoff=1e6)
    s.push(HandFrame(0, {"left": HandPose(np.zeros((24, 3))), "right": HandPose(np.ones((24, 3)))}, "a"))
    f = s.sample(0)
    assert f.hands["left"].finger_pose[0, 0] == 0 and f.hands["right"].finger_pose[0, 0] == pytest.approx(1)
    s.push(HandFrame(-5, {"left": HandPose(np.ones((24, 3)))}, "b"))  # new source: old clock doesn't apply
    assert set(s.sample(-5).hands) == {"left"}


def test_for_rate():
    slow, fast = SmoothingConfig.for_rate(5), SmoothingConfig.for_rate(200)
    assert fast.min_cutoff > slow.min_cutoff
    assert slow.hold_ms >= 600
