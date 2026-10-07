"""SOMA hand forward pass: HandPose -> mesh. The one place pose parameters become vertices (conventions in types.py).

Wraps py-soma-x's `SOMAHandLayer` (one per side, SOMA identity backend). Runs on CPU: Warp's skinning kernel poses a
2,859-vertex hand in ~0.3 ms there, well under a GPU round trip. Warp compiles its kernels on first use and caches
them (~/Library/Caches/warp), and the model assets download from Hugging Face on first use; both make the very first
start slow, later ones take a second or two.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hand_viewer.core.types import NUM_SHAPE, SIDES, HandPose, Side

LOD = "mid"   # 2,859 vertices per hand; "low" is 718


@dataclass
class HandMesh:
    vertices: np.ndarray  # (V, 3) float32
    joints: np.ndarray    # (25, 3) float32, SOMA_JOINTS order, same frame as vertices


def quiet_soma() -> None:
    """Warp prints a banner and a line per loaded module, and every layer re-checks its Hugging Face assets with a
    progress bar; keep the terminal for our own messages."""
    import warp
    from huggingface_hub.utils import disable_progress_bars

    warp.config.quiet = True
    warp.config.log_level = warp.LOG_WARNING
    disable_progress_bars()
    logging.getLogger("soma").setLevel(logging.WARNING)
    # PyTorch flags sparse CSR tensors (used by SOMA's skeleton transfer) as beta on every layer it builds.
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta", category=UserWarning)


def soma_layer(side: Side, **kwargs):
    """A CPU SOMAHandLayer with the output that py-soma-x prints during setup swallowed."""
    from soma.hand import SOMAHandLayer

    quiet_soma()
    with contextlib.redirect_stdout(io.StringIO()):
        layer = SOMAHandLayer(hand_type=side, device="cpu", lod=LOD, **kwargs)
    return layer.requires_grad_(False)


def wrist_frame(layer) -> np.ndarray:
    """(3, 3) rest orientation O of the wrist joint. SOMAHandLayer takes every T-pose-relative rotation q (wrist and
    fingers alike) in that tilted frame: the bone turns by O^T q O in model axes. HandPose rotations are plain
    model-axis rotations R (core/types.py), so q = O R O^T. The same O for both sides (tests/test_hand_model.py)."""
    return layer._t_pose_orient[0].numpy().astype(np.float64)


def to_layer(o: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """HandPose rotations (..., 3) -> SOMAHandLayer's (O R O^T)."""
    m = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    return Rotation.from_matrix(o @ m @ o.T).as_rotvec().reshape(rotvec.shape).astype(np.float32)


class HandModel:
    def __init__(self):
        self._layers = {side: soma_layer(side) for side in SIDES}
        self._wrist_frame = {side: wrist_frame(layer) for side, layer in self._layers.items()}
        self._shape: dict[Side, bytes | None] = {}
        self._faces = {}
        for side, layer in self._layers.items():
            f = np.ascontiguousarray(layer.faces.cpu().numpy().astype(np.int32))
            f.setflags(write=False)
            self._faces[side] = f
            self._prepare(side, None)

    @property
    def num_vertices(self) -> int:
        return int(self._faces["right"].max()) + 1

    def faces(self, side: Side) -> np.ndarray:
        """(F, 3) int32, outward-facing winding for `side`."""
        return self._faces[side]

    def _prepare(self, side: Side, shape: np.ndarray | None) -> None:
        """Cache the identity (rest shape + skeleton) for `shape`; a no-op when it's unchanged (the usual case:
        smoothing freezes shape per hand)."""
        key = None if shape is None else np.asarray(shape, np.float32).tobytes()
        if side in self._shape and self._shape[side] == key:
            return
        coeffs = torch.zeros(1, NUM_SHAPE) if shape is None else torch.from_numpy(np.asarray(shape, np.float32))[None]
        with torch.inference_mode():
            self._layers[side].prepare_identity(coeffs)
        self._shape[side] = key

    def forward(self, side: Side, pose: HandPose, *, default_orient: np.ndarray | None = None,
                anchor: np.ndarray | None = None) -> HandMesh:
        """Mesh for one hand. `wrist_orient` None -> `default_orient` (zeros if None). `wrist_position` None ->
        `anchor` (origin if None)."""
        return self.forward_many(side, [pose], default_orient=default_orient, anchor=anchor)[0]

    def forward_many(self, side: Side, poses: Sequence[HandPose], *, default_orient: np.ndarray | None = None,
                     anchor: np.ndarray | None = None) -> list[HandMesh]:
        """Batched `forward` for poses of one side that share a shape (all None, or equal)."""
        n = len(poses)
        if n == 0:
            return []
        shapes = {None if p.shape is None else p.shape.tobytes() for p in poses}
        if len(shapes) > 1:
            return [self.forward(side, p, default_orient=default_orient, anchor=anchor) for p in poses]
        self._prepare(side, poses[0].shape)
        orient0 = np.zeros(3, np.float32) if default_orient is None else np.asarray(default_orient, np.float32)
        anchor0 = np.zeros(3, np.float32) if anchor is None else np.asarray(anchor, np.float32)
        rot = np.empty((n, 25, 3), np.float32)
        transl = np.empty((n, 3), np.float32)
        for i, p in enumerate(poses):
            rot[i, 0] = orient0 if p.wrist_orient is None else p.wrist_orient
            rot[i, 1:] = p.finger_pose
            transl[i] = anchor0 if p.wrist_position is None else p.wrist_position
        rot = to_layer(self._wrist_frame[side], rot)
        with torch.inference_mode():
            out = self._layers[side].pose(torch.from_numpy(rot), global_translation=torch.from_numpy(transl))
        verts, joints = out.vertices.numpy(), out.joints.numpy()
        return [HandMesh(verts[i], joints[i]) for i in range(n)]
