"""HandModel and the contract's SOMA conventions (core/types.py): rest pose, left/right symmetry, plain wrist
rotation, rest-frame finger axes, wrist placement."""

import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_viewer.core.hand_model import HandMesh, HandModel
from hand_viewer.core.types import SIDES, SOMA_JOINTS, HandPose

J = {name: i for i, name in enumerate(SOMA_JOINTS)}


@pytest.fixture(scope="module")
def model():
    return HandModel()


def flat(**kw) -> HandPose:
    return HandPose(finger_pose=np.zeros((24, 3)), **kw)


def random_pose(seed, **kw) -> HandPose:
    rng = np.random.default_rng(seed)
    return HandPose(rng.normal(0, 0.3, (24, 3)), rng.normal(0, 1, 3), **kw)


def signed_volume(v, f):
    """Positive when triangle normals point outward (divergence theorem); the open wrist barely matters."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


@pytest.mark.parametrize("side", SIDES)
def test_rest_hand(model, side):
    m = model.forward(side, flat())
    assert isinstance(m, HandMesh)
    assert m.vertices.shape == (model.num_vertices, 3) == (2859, 3) and m.joints.shape == (25, 3)
    assert m.vertices.dtype == np.float32
    np.testing.assert_allclose(m.joints[J["Wrist"]], 0, atol=1e-6)
    # Flat: each finger's knuckle -> tip joints are nearly collinear, about a hand length (16-20 cm) from the wrist.
    for f in ("Index", "Middle", "Ring", "Pinky"):
        d = np.diff(m.joints[J[f + "2"]: J[f + "End"] + 1], axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        assert (d[1:] @ d[0] > 0.95).all(), f
    assert 0.16 < np.linalg.norm(m.joints[J["MiddleEnd"]]) < 0.2


@pytest.mark.parametrize("side", SIDES)
def test_faces_outward(model, side):
    f = model.faces(side)
    assert f.dtype == np.int32 and f.shape[1] == 3 and f.max() == model.num_vertices - 1
    assert signed_volume(model.forward(side, random_pose(0)).vertices, f) > 0


def test_left_is_right_reflected_through_wrist(model):
    p = random_pose(1)
    left, right = model.forward("left", p), model.forward("right", p)
    np.testing.assert_allclose(left.vertices, -right.vertices, atol=1e-4)
    np.testing.assert_allclose(left.joints, -right.joints, atol=1e-4)


@pytest.mark.parametrize("side", SIDES)
def test_wrist_orient_is_a_plain_rotation(model, side):
    r = np.array([0.3, -0.8, 0.5])
    rest = model.forward(side, flat()).vertices
    got = model.forward(side, flat(wrist_orient=r)).vertices
    np.testing.assert_allclose(got, rest @ Rotation.from_rotvec(r).as_matrix().T, atol=1e-5)


@pytest.mark.parametrize("side", SIDES)
def test_finger_rotation_is_about_rest_model_axes(model, side):
    r = np.array([0.1, -0.9, 0.2])
    fp = np.zeros((24, 3))
    fp[J["Index2"] - 1] = r   # finger_pose rows are joints 1..24
    rest, bent = model.forward(side, flat()).joints, model.forward(side, HandPose(fp)).joints
    knuckle = rest[J["Index2"]]
    np.testing.assert_allclose(bent[J["IndexEnd"]] - knuckle,
                               Rotation.from_rotvec(r).apply(rest[J["IndexEnd"]] - knuckle), atol=1e-5)


@pytest.mark.parametrize("side", SIDES)
def test_anchor_and_default_orient(model, side):
    anchor = np.array([0.1, -0.2, 0.5], np.float32)
    np.testing.assert_allclose(model.forward(side, flat(), anchor=anchor).joints[0], anchor, atol=1e-6)
    t = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(model.forward(side, flat(wrist_position=t), anchor=anchor).vertices,
                               model.forward(side, flat()).vertices + t, atol=1e-5)  # wrist_position wins
    rot = np.array([0.0, 0.0, np.pi / 2])
    a = model.forward(side, flat(), default_orient=rot).vertices
    np.testing.assert_allclose(a, model.forward(side, flat(wrist_orient=rot)).vertices, atol=1e-6)
    c = model.forward(side, flat(wrist_orient=np.zeros(3)), default_orient=rot).vertices
    np.testing.assert_allclose(c, model.forward(side, flat()).vertices, atol=1e-6)  # only when the pose has none


def test_shape_and_batching(model):
    shape = np.zeros(20)
    shape[0] = 2.0
    plain, shaped = model.forward("right", flat()), model.forward("right", flat(shape=shape))
    assert np.abs(plain.vertices - shaped.vertices).max() > 1e-3
    poses = [random_pose(i, wrist_position=[0, 0, 0.5]) for i in range(4)]
    mixed = poses[:2] + [random_pose(9, shape=shape)]
    for batch in (poses, mixed):
        for p, m in zip(batch, model.forward_many("left", batch)):
            np.testing.assert_allclose(m.vertices, model.forward("left", p).vertices, atol=1e-5)
    # The cached identity follows the pose: back to the mean hand after a shaped one.
    np.testing.assert_allclose(model.forward("right", flat()).vertices, plain.vertices, atol=1e-6)


def test_fast(model):
    pose = random_pose(2, wrist_position=[0, 0, 0.5])
    for _ in range(20):
        model.forward("left", pose)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        model.forward("left", pose)
    per_hand = (time.perf_counter() - t0) / n
    print(f"forward: {per_hand * 1e3:.3f} ms/hand")
    assert per_hand < 5e-3  # ~0.3 ms on an M4 Pro; loose so a busy machine doesn't flake
