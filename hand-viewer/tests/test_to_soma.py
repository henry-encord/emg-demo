"""ManoToSoma on real WiLoR output: the SOMA hands it produces must land on WiLoR's MANO meshes, for both sides."""

import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from hand_viewer.core.hand_model import HandModel
from hand_viewer.core.types import SIDES
from hand_viewer.mano.model import DEFAULT_MODEL_PATH, ManoParams

NPZ = Path(__file__).resolve().parents[2] / "encord-scene/out/sub-P001Fer_task-Glue_ep-005/mano.npz"

pytestmark = [pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="MANO_RIGHT.pkl not downloaded"),
              pytest.mark.skipif(not NPZ.exists(), reason=f"{NPZ} not found")]


@pytest.fixture(scope="module")
def converter():
    from hand_viewer.mano.to_soma import ManoToSoma

    return ManoToSoma()


@pytest.fixture(scope="module")
def model():
    return HandModel()


@pytest.fixture(scope="module")
def wilor():
    return dict(np.load(NPZ))


def rows(wilor, side, n=20):
    idx = np.flatnonzero(wilor[f"{side}_score"] > 0)
    return idx[np.linspace(0, len(idx) - 1, n).astype(int)]


def params(wilor, side, i) -> ManoParams:
    return ManoParams(wilor[f"{side}_hand_pose"][i], wilor[f"{side}_global_orient"][i], wilor[f"{side}_cam_t"][i],
                      wilor[f"{side}_betas"][i])


@pytest.mark.parametrize("side", SIDES)
def test_lands_on_wilor_mesh(converter, model, wilor, side):
    """Distances from SOMA vertices to WiLoR's MANO vertices, camera frame. Measured: median ~5.5 mm, p90 ~10 mm,
    most of it the SOMA mean hand vs the person's MANO shape (the fit itself is ~3 mm). A wrong side convention
    or wrist frame is off by centimetres."""
    medians, p90s = [], []
    for i in rows(wilor, side):
        pose = converter.convert(side, [params(wilor, side, i)], [0.8])[0]
        assert pose.frame == "camera" and pose.confidence == pytest.approx(0.8) and pose.shape is None
        got = model.forward(side, pose).vertices
        d, _ = cKDTree(wilor[f"{side}_vertices"][i] + wilor[f"{side}_cam_t"][i]).query(got)
        medians.append(np.median(d))
        p90s.append(np.percentile(d, 90))
    assert np.mean(medians) < 0.008 and np.mean(p90s) < 0.015, (np.mean(medians), np.mean(p90s))


def test_batch_matches_single(converter, wilor):
    side = "left"
    idx = rows(wilor, side, 6)
    batch = converter.convert(side, [params(wilor, side, i) for i in idx])
    for i, b in zip(idx, batch):
        s = converter.convert(side, [params(wilor, side, i)])[0]
        # Single calls keep slightly different betas prepared (BETAS_TOLERANCE), so allow a little.
        np.testing.assert_allclose(s.finger_pose, b.finger_pose, atol=0.05)
        np.testing.assert_allclose(s.wrist_position, b.wrist_position, atol=2e-3)


def test_mixed_sides_match_per_side(converter, wilor):
    both = np.flatnonzero((wilor["left_score"] > 0) & (wilor["right_score"] > 0))[::50][:5]
    for i in both:
        mixed = converter.convert_many(["left", "right"], [params(wilor, "left", i), params(wilor, "right", i)])
        for side, got in zip(("left", "right"), mixed):
            want = converter.convert(side, [params(wilor, side, i)])[0]
            np.testing.assert_allclose(got.finger_pose, want.finger_pose, atol=0.05)
            np.testing.assert_allclose(got.wrist_orient, want.wrist_orient, atol=0.05)
            np.testing.assert_allclose(got.wrist_position, want.wrist_position, atol=2e-3)


def test_fast(converter, wilor):
    side = "right"
    ps = [params(wilor, side, i) for i in rows(wilor, side, 30)]
    converter.convert(side, ps[:1])
    t0 = time.perf_counter()
    for p in ps:
        converter.convert(side, [p])
    per_hand = (time.perf_counter() - t0) / len(ps)
    print(f"convert: {per_hand * 1e3:.1f} ms/hand")
    assert per_hand < 0.03  # ~5 ms on an M4 Pro (~16 ms more whenever the betas move enough to re-prepare)


def test_both_hands_batched_is_faster(converter, wilor):
    both = np.flatnonzero((wilor["left_score"] > 0) & (wilor["right_score"] > 0))[:30]
    pairs = [[params(wilor, "left", i), params(wilor, "right", i)] for i in both]
    converter.convert_many(["left", "right"], pairs[0])

    def per_frame(f):
        t0 = time.perf_counter()
        for p in pairs:
            f(p)
        return (time.perf_counter() - t0) / len(pairs)

    separate = per_frame(lambda p: (converter.convert("left", p[:1]), converter.convert("right", p[1:])))
    batched = per_frame(lambda p: converter.convert_many(["left", "right"], p))
    print(f"both hands: separate {separate * 1e3:.1f} ms, batched {batched * 1e3:.1f} ms")
    assert batched < 0.85 * separate
