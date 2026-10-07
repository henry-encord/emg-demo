"""MANO forward pass: ManoParams -> mesh. Only WiLoR and replayed mano.npz files speak MANO; `to_soma` converts their
output to the app's SOMA contract (core/types.py), so nothing past the pose sources sees these types.

MANO conventions (validated against WiLoR output in encord-scene/out/*/mano.npz to ~1e-7 m):
- Rotations are axis-angle (rotation vectors). `hand_pose` zeros is a flat open hand (MANO without the pose mean,
  i.e. `smplx.MANOLayer` / `smplx.MANO(use_pca=False, flat_hand_mean=True)`).
- Only the right-hand MANO model is used. Right-hand params go straight in. Left-hand params use WiLoR's mirrored
  convention: negate the y and z components of `global_orient` and every `hand_pose` row, run the right-hand model,
  then negate vertex x (and reverse face winding). Translation is added after mirroring.
- `transl` is added to the MANO output vertices (like WiLoR's cam_t / smplx's transl). It is NOT the wrist position:
  MANO's root joint sits ~10 cm from the MANO origin.

Reproduces WiLoR's MANO layer (smplx `MANOLayer`, no pose mean, MANO 16 joints + 5 fingertip vertices in OpenPose
order) without importing wilor_mini. Runs on CPU in float32: a hand is ~800 vertices, so a GPU round trip would cost
more than the maths.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from hand_viewer.core.types import Side

DEFAULT_MODEL_PATH = Path.home() / ".cache/wilor-mini/pretrained_models/MANO_RIGHT.pkl"
# Fingertip vertices (thumb, index, middle, ring, pinky), as smplx.vertex_ids["mano"] and WiLoR use.
TIP_VERTICES = (744, 320, 443, 554, 671)
# MANO joints (16) + tips (5) -> OpenPose hand order; copied from wilor_mini/models/mano_wrapper.py.
MANO_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
# Left hands are the right model mirrored in x: an axis-angle r maps to (r_x, -r_y, -r_z) under that reflection.
_MIRROR_ROT = torch.tensor([1.0, -1.0, -1.0])
_MIRROR_POS = np.array([-1.0, 1.0, 1.0], np.float32)


@dataclass(frozen=True)
class ManoParams:
    """One hand's MANO parameters, in the side's own convention (see above)."""

    hand_pose: np.ndarray                    # (15, 3) axis-angle per finger joint; 0 = flat
    global_orient: np.ndarray | None = None  # (3,) axis-angle wrist rotation
    transl: np.ndarray | None = None         # (3,) metres, added to the vertices
    betas: np.ndarray | None = None          # (10,) shape; None = mean hand

    def __post_init__(self):
        for name, shape in (("hand_pose", (15, 3)), ("global_orient", (3,)), ("transl", (3,)), ("betas", (10,))):
            a = getattr(self, name)
            if a is not None:
                object.__setattr__(self, name, np.asarray(a, np.float32).reshape(shape))


@dataclass
class HandMesh:
    vertices: np.ndarray  # (778, 3) float32
    joints: np.ndarray    # (21, 3) float32, OpenPose order (MANO 16 + 5 tips), same frame as vertices


class ManoModel:
    def __init__(self, model_path: Path | None = None):
        import smplx  # deferred: smplx pulls in a lot, and only this module needs it
        from smplx.lbs import lbs

        path = Path(model_path or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"MANO model not found at {path} (WiLoR-mini downloads it on first run)")
        # The pickle holds chumpy objects, which trip deprecation warnings on load, and smplx prints a "only 10 shape
        # coefficients" notice; both are harmless.
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            layer = smplx.MANO(str(path), is_rhand=True, use_pca=False, flat_hand_mean=True)
        self._lbs = lbs
        # Keep only the buffers lbs needs; calling lbs directly skips smplx's per-call bookkeeping.
        self._buffers = tuple(getattr(layer, k).detach().to(torch.float32).contiguous() for k in
                              ("v_template", "shapedirs", "posedirs", "J_regressor"))
        self._parents = layer.parents.detach()
        self._weights = layer.lbs_weights.detach().to(torch.float32).contiguous()
        self._num_betas = layer.num_betas
        self._tips = torch.tensor(TIP_VERTICES, dtype=torch.long)
        self._joint_map = torch.tensor(MANO_TO_OPENPOSE, dtype=torch.long)
        faces = np.asarray(layer.faces, np.int32)
        # Mirroring x flips triangle orientation, so the left hand needs reversed winding to keep normals outward.
        self._faces = {"right": _readonly(faces), "left": _readonly(faces[:, ::-1])}

    def faces(self, side: Side) -> np.ndarray:
        """(1538, 3) int32, outward-facing winding for `side`."""
        return self._faces[side]

    def forward(self, side: Side, pose: ManoParams, *, default_orient: np.ndarray | None = None,
                anchor: np.ndarray | None = None) -> HandMesh:
        """Mesh for one hand. `global_orient` None -> `default_orient` (same side convention as the pose; zeros if
        None). `transl` None and `anchor` given -> translated so the root joint (joints[0], the wrist) is on it."""
        return self.forward_many(side, [pose], default_orient=default_orient, anchor=anchor)[0]

    def forward_many(self, side: Side, poses: Sequence[ManoParams], *, default_orient: np.ndarray | None = None,
                     anchor: np.ndarray | None = None) -> list[HandMesh]:
        """Batched `forward` for many poses of one side (e.g. a whole replay), one lbs call."""
        n = len(poses)
        if n == 0:
            return []
        orient0 = np.zeros(3, np.float32) if default_orient is None else np.asarray(default_orient, np.float32)
        full = np.empty((n, 16, 3), np.float32)
        betas = np.zeros((n, self._num_betas), np.float32)
        transl = np.zeros((n, 3), np.float32)
        anchored = np.zeros(n, bool)
        for i, p in enumerate(poses):
            full[i, 0] = orient0 if p.global_orient is None else p.global_orient
            full[i, 1:] = p.hand_pose
            if p.betas is not None:
                betas[i] = p.betas
            if p.transl is not None:
                transl[i] = p.transl
            else:
                anchored[i] = anchor is not None
        pose_t = torch.from_numpy(full)
        if side == "left":
            pose_t = pose_t * _MIRROR_ROT
        with torch.inference_mode():
            verts, joints = self._lbs(torch.from_numpy(betas), pose_t.reshape(n, 48), *self._buffers,
                                      self._parents, self._weights, pose2rot=True)
            joints = torch.cat([joints, verts[:, self._tips]], dim=1)[:, self._joint_map]
        verts, joints = verts.numpy(), joints.numpy()
        if side == "left":
            verts, joints = verts * _MIRROR_POS, joints * _MIRROR_POS
        if anchored.any():
            transl[anchored] = np.asarray(anchor, np.float32) - joints[anchored, 0]
        verts = verts + transl[:, None]
        joints = joints + transl[:, None]
        return [HandMesh(verts[i], joints[i]) for i in range(n)]


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a
