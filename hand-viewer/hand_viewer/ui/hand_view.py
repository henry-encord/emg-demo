"""Right pane: the 3D hand render. `HandRenderer` is the seam (pyqtgraph now, maybe pyvistaqt later).

GL scene axes are pyqtgraph's native z-up: X right, Y forward (away from the viewer), Z up. Camera-frame poses
(OpenCV: x right, y down, z forward) map with X=x, Y=z, Z=-y, so the video camera sits at the GL origin looking down
+Y. World-frame poses are assumed z-up already and get only the optional "zero heading" yaw.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QWidget
from pyqtgraph import Vector
from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader
from scipy.spatial.transform import Rotation

from hand_viewer.core.mano import ManoModel
from hand_viewer.core.types import SIDES, HandFrame, HandPose, Side
from hand_viewer.ui import theme

# Camera (OpenCV) -> GL scene. Rows map x_cam, y_cam, z_cam into X, Y, Z: a proper rotation (det +1).
CAM_TO_GL = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)
# Hands without a position: root joints this far apart, this far in front of the camera (camera frame, metres).
ANCHOR_HALF_SPACING = 0.12
ANCHOR_DEPTH = 0.5
FLOOR_BELOW = 0.35   # grid this far below the camera
COLORS = {side: theme.rgba(c) for side, c in theme.HAND_COLORS.items()}   # chart-1 olive, chart-2 blue

# pyqtgraph's built-in "shaded" lights from behind the scene (direction (1,-1,-1) in view space), so surfaces facing
# the viewer come out near-black. This is the same program with a key light over the viewer's shoulder plus a fill.
HEADLIGHT = ShaderProgram("handHeadlight", [
    VertexShader("""
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """),
    FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            vec3 n = normalize(v_normal);
            float key = max(dot(n, normalize(vec3(-0.4, 0.6, 1.0))), 0.0);
            float fill = max(dot(n, normalize(vec3(0.6, -0.3, 0.5))), 0.0);
            gl_FragColor = vec4(v_color.rgb * (0.25 + 0.65 * key + 0.25 * fill), v_color.a);
        }
    """),
])


class HandRenderer(ABC):
    widget: QWidget

    @abstractmethod
    def update(self, hand_frame: HandFrame | None, mano: ManoModel) -> None: ...

    @abstractmethod
    def set_camera_view(self, hfov_deg: float) -> None:
        """Put the 3D camera where the video camera is, so the render overlays the video's viewpoint."""

    @abstractmethod
    def reset_view(self) -> None: ...

    @abstractmethod
    def zero_heading(self) -> None:
        """For frame="world" poses: rotate about the vertical so the current hand faces away from the viewer."""


def _anchor(side: Side) -> np.ndarray:
    """Camera-frame anchor for an unpositioned hand: left hand on the left of the image, as in egocentric video."""
    return np.array([-ANCHOR_HALF_SPACING if side == "left" else ANCHOR_HALF_SPACING, 0.0, ANCHOR_DEPTH], np.float32)


def _basis(joints: np.ndarray) -> np.ndarray:
    """Columns: fingers direction (wrist -> middle MCP), thumb side (pinky MCP -> index MCP), their cross."""
    up = joints[9] - joints[0]
    up = up / np.linalg.norm(up)
    lat = joints[5] - joints[17]
    lat = lat - lat.dot(up) * up
    lat = lat / np.linalg.norm(lat)
    return np.stack([up, lat, np.cross(up, lat)], axis=1)


def _yaw(deg_rad: float) -> np.ndarray:
    c, s = np.cos(deg_rad), np.sin(deg_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


class PyqtgraphHandRenderer(HandRenderer):
    def __init__(self, parent=None):
        self.widget = view = gl.GLViewWidget(parent)
        fmt = view.format()
        fmt.setSamples(4)   # MSAA on this widget only; no global QSurfaceFormat
        view.setFormat(fmt)
        view.setBackgroundColor(theme.VIEWPORT)
        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.1, 0.1)
        grid.setColor(theme.GRID)
        grid.translate(0, ANCHOR_DEPTH, -FLOOR_BELOW)
        view.addItem(grid)
        view.addItem(gl.GLAxisItem(size=Vector(0.05, 0.05, 0.05)))   # marks the video camera's position
        self._meshes: dict[Side, gl.GLMeshItem] = {}
        self._mesh_data: dict[Side, gl.MeshData] = {}
        self._default_orient: dict[tuple[Side, str], np.ndarray] = {}
        self._mano: ManoModel | None = None
        self._yaw = 0.0
        self._last: HandFrame | None = None
        self.reset_view()

    # ------------------------------------------------------------------------------------------ view

    def reset_view(self) -> None:
        self.widget.setCameraParams(center=Vector(0, ANCHOR_DEPTH, 0), distance=0.9, elevation=20, azimuth=-110,
                                    fov=60)

    def set_camera_view(self, hfov_deg: float) -> None:
        # pyqtgraph's camera sits at center + distance * (cos e cos a, cos e sin a, sin e) and its fov is horizontal,
        # so azimuth -90, elevation 0 and center (0, d, 0) puts it at the origin looking down +Y. Orbiting then
        # pivots about a point at the hands' depth.
        self.widget.setCameraParams(center=Vector(0, ANCHOR_DEPTH, 0), distance=ANCHOR_DEPTH, elevation=0,
                                    azimuth=-90, fov=hfov_deg)

    def zero_heading(self) -> None:
        f, mano = self._last, self._mano
        if f is None or mano is None:
            return
        side, pose = next(((s, p) for s, p in sorted(f.hands.items(), key=lambda kv: kv[0] != "right")
                           if p.frame == "world" and p.global_orient is not None), (None, None))
        if pose is None:
            return
        j = mano.forward(side, pose).joints
        d = j[9] - j[0]
        # Current heading of the fingers in the horizontal plane; rotate it onto +Y (away from the viewer).
        self._yaw = np.pi / 2 - float(np.arctan2(d[1], d[0]))

    # ------------------------------------------------------------------------------------------ hands

    def _orient(self, mano: ManoModel, side: Side, frame: str) -> np.ndarray:
        """global_orient for a hand that has none: palm towards the viewer, fingers up, thumb outwards.

        Measured from the rest pose rather than hard-coded, so it holds whatever MANO's canonical axes are. With
        forward()'s mirroring, the left output is R(p) applied to the left rest mesh, so the same solve works per side.
        """
        key = (side, frame)
        if key not in self._default_orient:
            rest = mano.forward(side, HandPose(hand_pose=np.zeros((15, 3)))).joints
            thumb = 1.0 if side == "right" else -1.0
            up = np.array([0, -1, 0] if frame == "camera" else [0, 0, 1], np.float32)
            target = np.stack([up, [thumb, 0, 0], np.cross(up, [thumb, 0, 0])], axis=1)
            r = target @ _basis(rest).T
            self._default_orient[key] = Rotation.from_matrix(r).as_rotvec().astype(np.float32)
        return self._default_orient[key]

    def _mesh_item(self, side: Side, mano: ManoModel) -> tuple[gl.GLMeshItem, gl.MeshData]:
        if side not in self._meshes or self._mano is not mano:
            if side in self._meshes:
                self.widget.removeItem(self._meshes[side])
            md = gl.MeshData(vertexes=np.zeros((778, 3), np.float32), faces=mano.faces(side))
            item = gl.GLMeshItem(meshdata=md, smooth=True, shader=HEADLIGHT, color=COLORS[side], glOptions="opaque")
            self.widget.addItem(item)
            self._meshes[side], self._mesh_data[side] = item, md
        return self._meshes[side], self._mesh_data[side]

    def update(self, hand_frame: HandFrame | None, mano: ManoModel) -> None:
        self._last = hand_frame
        hands = hand_frame.hands if hand_frame is not None else {}
        for side in SIDES:
            pose = hands.get(side)
            if pose is None:
                if side in self._meshes:
                    self._meshes[side].setVisible(False)
                continue
            item, md = self._mesh_item(side, mano)
            md.setVertexes(self._vertices(side, pose, mano))
            item.meshDataChanged()
            item.setVisible(True)
        self._mano = mano

    def _vertices(self, side: Side, pose: HandPose, mano: ManoModel) -> np.ndarray:
        orient = self._orient(mano, side, pose.frame)
        if pose.frame == "camera":
            v = mano.forward(side, pose, default_orient=orient, anchor=_anchor(side)).vertices
            return v @ CAM_TO_GL.T
        anchor = _anchor(side) @ CAM_TO_GL.T   # same on-screen spot, expressed in the z-up frame
        mesh = mano.forward(side, pose, default_orient=orient, anchor=anchor)
        if self._yaw == 0.0:
            return mesh.vertices
        # Yaw about the world origin when the source gives positions (keeps the hands' layout), else the wrist.
        pivot = np.zeros(3, np.float32) if pose.transl is not None else anchor
        return (mesh.vertices - pivot) @ _yaw(self._yaw).T + pivot
