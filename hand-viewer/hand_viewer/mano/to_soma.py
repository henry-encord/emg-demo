"""MANO -> SOMA: turns WiLoR's MANO parameters into the app's HandPose (SOMA joint rotations, see core/types.py).

There's no closed-form map between the skeletons (MANO has 16 joints, SOMA 25 with metacarpals and different joint
placement), so this goes through the mesh, as py-soma-x's own tools/hand/mano2soma.py does:
1. pose the MANO mesh with the right-hand model (left-hand params get their y and z negated first, as in ManoModel),
2. move it onto SOMA's hand topology with SOMA-X's barycentric MANO->SOMA map, wrist at the origin,
3. fit SOMA joint rotations to it with `PoseInversion` (analytical warm start + Gauss-Newton).
That costs ~7 ms for one hand on CPU, ~9 ms for both in one `convert_many`, with a mean vertex error of ~1.5-2 mm against the MANO mesh.

Left hands: there's no MANO_LEFT.pkl, so step 3 always fits the right SOMA hand to the right-model mesh. The real
left hand is that mesh mirrored in x. SOMA's left mesh for pose p is the right mesh for p reflected through the wrist
(v -> -v), and mirroring x after reflecting through the origin is a 180-degree turn about x. So the left pose is the
right fit with the wrist rotation turned 180 degrees about x, and the wrist position mirrored in x.

Hand shape is not converted: the SOMA mean hand is rendered (shape None). MANO betas are only used to fit the pose.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hand_viewer.core.hand_model import soma_layer, wrist_frame
from hand_viewer.core.types import HandPose, Side
from hand_viewer.mano.model import DEFAULT_MODEL_PATH, ManoModel, ManoParams

_MIRROR_X = np.array([-1.0, 1.0, 1.0], np.float32)
_MIRROR_ROT = np.array([1.0, -1.0, -1.0], np.float32)   # an axis-angle under x -> -x
_TURN_X = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
# Preparing the fit for new betas costs ~16 ms, three times the fit itself, and WiLoR's betas jitter every frame.
# Betas only shape the mesh being fitted, so live-sized calls (up to LIVE_BATCH hands) keep the prepared betas until
# one moves this far.
BETAS_TOLERANCE = 0.25
LIVE_BATCH = 2


class ManoToSoma:
    """Not thread-safe; give each worker its own instance."""

    def __init__(self, mano: ManoModel | None = None, model_path: Path | None = None, lie_iters: int = 1):
        from soma.fitting.pose_inversion import PoseInversion
        from soma.geometry.rig_utils import apply_joint_orient_local, remove_joint_orient_local

        path = Path(model_path or DEFAULT_MODEL_PATH)
        self.mano = mano or ManoModel(path)
        self.lie_iters = lie_iters
        # Fitting layer: SOMA skeleton, shaped by MANO betas, accepting MANO-topology meshes.
        self._fit_layer = soma_layer("right", identity_model_type="mano", identity_model_kwargs={"model_path": str(path)})
        self._inv = PoseInversion(self._fit_layer, low_lod=False)
        self._apply, self._remove = apply_joint_orient_local, remove_joint_orient_local
        # T-pose joint orients turn the fit's absolute local rotations into the contract's T-pose-relative ones.
        left = soma_layer("left")
        self._orient = {"right": (self._fit_layer._t_pose_orient, self._fit_layer._t_pose_orient_parent_T),
                        "left": (left._t_pose_orient, left._t_pose_orient_parent_T)}
        self._wrist_frame = {"right": wrist_frame(self._fit_layer), "left": wrist_frame(left)}
        self._betas: np.ndarray | None = None

    def convert(self, side: Side, params: Sequence[ManoParams], confidence: Sequence[float] | None = None
                ) -> list[HandPose]:
        """Camera-frame HandPoses for MANO params of one side (batched; params in the side's own convention)."""
        return self.convert_many([side] * len(params), params, confidence)

    def convert_many(self, sides: Sequence[Side], params: Sequence[ManoParams],
                     confidence: Sequence[float] | None = None) -> list[HandPose]:
        """`convert` for hands of either side in one batch (e.g. both hands of a live frame: one fit instead of two,
        since every hand is fitted with the right model). `sides[i]` is the side of `params[i]`."""
        n = len(params)
        if n == 0:
            return []
        left = np.array([s == "left" for s in sides])
        right_model = [replace(p, transl=None) for p in params]
        right_model = [replace(p, hand_pose=p.hand_pose * _MIRROR_ROT,
                               global_orient=None if p.global_orient is None else p.global_orient * _MIRROR_ROT)
                       if is_left else p for p, is_left in zip(right_model, left)]
        meshes = self.mano.forward_many("right", right_model)
        verts = np.stack([m.vertices for m in meshes])
        wrist = np.stack([m.joints[0] for m in meshes])
        betas = np.stack([np.zeros(10, np.float32) if p.betas is None else p.betas for p in params])
        with torch.inference_mode():
            target = self._fit_layer.identity_model._to_soma_interp(torch.from_numpy(verts - wrist[:, None]))
            self._prepare(betas)
            fit = self._inv.fit(target, body_iters=0, finger_iters=0, full_iters=1, lie_iters=self.lie_iters)
            rel = self._remove(fit["rotations"], *self._orient["right"])
            if left.any():
                li = torch.from_numpy(np.flatnonzero(left))
                absolute = self._apply(rel[li], *self._orient["left"])
                absolute[:, 0] = _TURN_X @ absolute[:, 0]
                rel[li] = self._remove(absolute, *self._orient["left"])
            root = fit["root_translation"].numpy()
        rel = rel.numpy().astype(np.float64)
        o = np.stack([self._wrist_frame[s] for s in sides])[:, None]
        rel = o.transpose(0, 1, 3, 2) @ rel @ o  # the layer's wrist-frame rotations -> plain model-axis rotations
        rotvec = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(n, 25, 3)
        position = wrist + root
        position[left] *= _MIRROR_X
        transl = np.stack([np.zeros(3, np.float32) if p.transl is None else p.transl for p in params])
        position = position + transl
        conf = [1.0] * n if confidence is None else list(confidence)
        return [HandPose(rotvec[i, 1:], rotvec[i, 0], position[i], None, "camera", float(conf[i])) for i in range(n)]

    def _prepare(self, betas: np.ndarray) -> None:
        old = self._betas
        if old is not None and old.shape == betas.shape and (
                np.array_equal(old, betas) or (len(betas) <= LIVE_BATCH and np.abs(old - betas).max() <= BETAS_TOLERANCE)):
            return
        self._inv.prepare_identity(torch.from_numpy(betas))
        self._betas = betas
