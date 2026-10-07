"""ManoModel must reproduce WiLoR's MANO output exactly; everything downstream trusts it."""

import time
from pathlib import Path

import numpy as np
import pytest

from hand_viewer.core.types import SIDES
from hand_viewer.mano.model import DEFAULT_MODEL_PATH, HandMesh, ManoModel
from hand_viewer.mano.model import ManoParams as HandPose

NPZ = Path(__file__).resolve().parents[2] / "encord-scene/out/sub-P001Fer_task-Glue_ep-005/mano.npz"

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="MANO_RIGHT.pkl not downloaded")


@pytest.fixture(scope="module")
def mano():
    return ManoModel()


@pytest.fixture(scope="module")
def wilor():
    if not NPZ.exists():
        pytest.skip(f"{NPZ} not found")
    return dict(np.load(NPZ))


def flat() -> HandPose:
    return HandPose(hand_pose=np.zeros((15, 3)))


def signed_volume(v, f):
    """Positive when triangle normals point outward (divergence theorem); MANO's wrist opening barely matters."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def test_flat_hand_is_template(mano):
    m = mano.forward("right", flat())
    assert isinstance(m, HandMesh)
    assert m.vertices.shape == (778, 3) and m.joints.shape == (21, 3)
    assert m.vertices.dtype == np.float32 and m.joints.dtype == np.float32
    v_template = mano._buffers[0].numpy()
    np.testing.assert_allclose(m.vertices, v_template, atol=1e-6)
    # Flat fingers: each non-thumb finger's 4 joints (MCP..tip, OpenPose order) are nearly collinear.
    for f in range(1, 5):
        d = np.diff(m.joints[1 + 4 * f: 5 + 4 * f], axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        assert (d[1:] @ d[0] > 0.9).all()


def test_left_flat_is_mirror_of_right(mano):
    r, lft = mano.forward("right", flat()), mano.forward("left", flat())
    np.testing.assert_allclose(lft.vertices, r.vertices * [-1, 1, 1], atol=1e-7)


@pytest.mark.parametrize("side", SIDES)
def test_reproduces_wilor(mano, wilor, side):
    idx = np.flatnonzero(wilor[f"{side}_score"] > 0)
    assert len(idx) >= 20
    idx = idx[np.linspace(0, len(idx) - 1, 20).astype(int)]
    for i in idx:
        kw = dict(global_orient=wilor[f"{side}_global_orient"][i], hand_pose=wilor[f"{side}_hand_pose"][i],
                  betas=wilor[f"{side}_betas"][i])
        want = wilor[f"{side}_vertices"][i]
        got = mano.forward(side, HandPose(**kw)).vertices
        np.testing.assert_allclose(got, want, atol=1e-4, err_msg=f"{side} frame {i}")
        cam_t = wilor[f"{side}_cam_t"][i]
        got = mano.forward(side, HandPose(**kw, transl=cam_t)).vertices
        np.testing.assert_allclose(got, want + cam_t, atol=1e-4, err_msg=f"{side} frame {i} + cam_t")


def test_forward_many_matches_forward(mano, wilor):
    idx = np.flatnonzero(wilor["left_score"] > 0)[:8]
    poses = [HandPose(hand_pose=wilor["left_hand_pose"][i], global_orient=wilor["left_global_orient"][i],
                      betas=wilor["left_betas"][i], transl=wilor["left_cam_t"][i]) for i in idx]
    for p, m in zip(poses, mano.forward_many("left", poses)):
        np.testing.assert_allclose(m.vertices, mano.forward("left", p).vertices, atol=1e-6)


@pytest.mark.parametrize("side", SIDES)
def test_faces_outward(mano, side):
    rng = np.random.default_rng(0)
    pose = HandPose(hand_pose=rng.normal(0, 0.2, (15, 3)), global_orient=rng.normal(0, 1, 3))
    f = mano.faces(side)
    assert f.shape == (1538, 3) and f.dtype == np.int32
    assert signed_volume(mano.forward(side, pose).vertices, f) > 0


def test_faces_left_is_reversed(mano):
    np.testing.assert_array_equal(mano.faces("left"), mano.faces("right")[:, ::-1])


@pytest.mark.parametrize("side", SIDES)
def test_anchor_and_default_orient(mano, side):
    anchor = np.array([0.1, -0.2, 0.5], np.float32)
    m = mano.forward(side, flat(), anchor=anchor)
    np.testing.assert_allclose(m.joints[0], anchor, atol=1e-6)
    # transl wins over anchor.
    t = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(mano.forward(side, HandPose(np.zeros((15, 3)), transl=t), anchor=anchor).vertices,
                               mano.forward(side, flat()).vertices + t, atol=1e-6)
    # default_orient applies only when the pose has no global_orient.
    rot = np.array([0.0, 0.0, np.pi / 2])
    a = mano.forward(side, flat(), default_orient=rot)
    b = mano.forward(side, HandPose(np.zeros((15, 3)), global_orient=rot))
    np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-7)
    c = mano.forward(side, HandPose(np.zeros((15, 3)), global_orient=np.zeros(3)), default_orient=rot)
    np.testing.assert_allclose(c.vertices, mano.forward(side, flat()).vertices, atol=1e-7)


def test_fast(mano):
    rng = np.random.default_rng(1)
    pose = HandPose(hand_pose=rng.normal(0, 0.2, (15, 3)), global_orient=rng.normal(0, 1, 3), transl=[0, 0, 0.5])
    for _ in range(20):
        mano.forward("left", pose)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        mano.forward("left", pose)
    per_hand = (time.perf_counter() - t0) / n
    print(f"forward: {per_hand * 1e3:.3f} ms/hand")
    assert per_hand < 5e-3  # target is < 1 ms; loose bound so a busy CI box doesn't flake
