"""Temporal smoothing of HandFrames: a One-Euro filter per hand, then render-delay interpolation.

Pose sources arrive at very different rates (WiLoR ~5 Hz with frame-to-frame jitter, an EMG model maybe 100+ Hz)
while the UI renders at ~60 Hz. `push` filters each sample as it arrives; `sample(t)` renders slightly in the past
(t - render delay) so there are usually two filtered samples to interpolate between, which turns a 5 Hz stream
into smooth motion at the cost of about one sample interval of extra latency.

Rotations (global_orient + 15 hand_pose joints) are filtered as unit quaternions in the tangent space around the
previous filtered value (the relative rotation's log), with a hemisphere check so q and -q are treated as the same
rotation. Translation is filtered as a plain vector. Betas are frozen to the per-component median of the first N
samples of a hand, since hand shape doesn't change and letting it float only adds jitter. None fields stay None.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hand_viewer.core.types import HandFrame, HandPose, Side, now_ns

# A new sample further than this from the previous one of the same hand starts a fresh track (no filtering or
# interpolation across the gap). Betas are re-estimated once a hand has been gone this long.
BETAS_RESET_NS = 2_000_000_000
MAX_AUTO_DELAY_MS = 400.0
# Intervals/latencies above this are gaps or foreign clocks, not the stream's rate, and aren't averaged in.
MAX_MEASURED_NS = 1_000_000_000
EMA_ALPHA = 0.1
# One-Euro's beta multiplies speed. Rotations are in rad/s; scale transl (m/s) so 10 cm/s counts like 1 rad/s,
# roughly the same visual motion for a hand, and one beta works for both.
TRANSL_SPEED_SCALE = 10.0
HISTORY = 64


@dataclass
class SmoothingConfig:
    enabled: bool = True
    min_cutoff: float = 1.0               # Hz; lower = smoother when still, more lag
    beta: float = 0.3                     # speed coefficient; higher = less lag when moving fast
    d_cutoff: float = 1.0                 # Hz; cutoff for the speed estimate
    render_delay_ms: float | None = None  # None = auto (about one measured pose interval plus arrival latency)
    hold_ms: float = 300                  # keep a lost hand this long, then drop it
    freeze_betas_after: int = 10          # median of first N frames per hand; 0 = never

    @classmethod
    def for_rate(cls, hz: float) -> SmoothingConfig:
        """Suggested settings for a source producing about `hz` samples/s. Slow, jittery sources (WiLoR) need
        heavier filtering and interpolation; fast ones (EMG) only need light denoising."""
        interval_ms = 1000.0 / max(hz, 1e-3)
        hold_ms = max(300.0, 3 * interval_ms)
        if hz < 15:
            return cls(min_cutoff=1.0, beta=0.3, hold_ms=hold_ms)
        if hz < 60:
            return cls(min_cutoff=2.0, beta=0.5, hold_ms=hold_ms)
        return cls(min_cutoff=4.0, beta=0.7, hold_ms=hold_ms)


# ---------------------------------------------------------------------------------------------- quaternions
# (w, x, y, z), vectorised over leading axes. Small and local so the 60 Hz path avoids scipy object overhead.


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(a/2)/a, with its Taylor limit near 0
    k = np.where(angle > 1e-8, np.sin(half) / np.maximum(angle, 1e-12), 0.5 - angle**2 / 48)
    return np.concatenate([np.cos(half), v * k], axis=-1)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    q = np.where(q[..., :1] < 0, -q, q)  # w >= 0 gives the shortest rotation (angle <= pi)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(s, q[..., :1])
    k = np.where(s > 1e-8, angle / np.maximum(s, 1e-12), 2.0)
    return q[..., 1:] * k


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def align_hemisphere(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Flip each q onto the hemisphere of ref (q and -q are the same rotation; blending needs the near one)."""
    return np.where(np.sum(q * ref, axis=-1, keepdims=True) < 0, -q, q)


def quat_log_rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation vector of a^-1 b (the step from a to b in a's tangent space)."""
    return rotvec_from_quat(quat_mul(quat_conj(a), align_hemisphere(b, a)))


def quat_step(a: np.ndarray, delta: np.ndarray) -> np.ndarray:
    q = quat_mul(a, quat_from_rotvec(delta))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def slerp(a: np.ndarray, b: np.ndarray, u: np.ndarray | float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if u.ndim:
        u = u[..., None]
    return quat_step(a, quat_log_rel(a, b) * u)


# ---------------------------------------------------------------------------------------------- One-Euro


def _alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class _Filtered:
    """One filtered sample of one hand (internal representation: quaternions, float64)."""

    t_ns: int
    rot: np.ndarray                 # (16, 4): row 0 global_orient (identity if absent), rows 1.. hand_pose
    has_orient: bool
    transl: np.ndarray | None
    betas: np.ndarray | None
    frame: str
    confidence: float


@dataclass
class _Track:
    """Filter state and recent filtered samples for one hand."""

    last: _Filtered | None = None
    rot_speed: np.ndarray = field(default_factory=lambda: np.zeros((16, 3)))
    transl_speed: np.ndarray = field(default_factory=lambda: np.zeros(3))
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    betas_seen: list = field(default_factory=list)
    betas_frozen: np.ndarray | None = None
    last_seen_ns: int | None = None


class PoseSmoother:
    """Not thread-safe: call `push` and `sample` from one thread (the session's UI timer)."""

    def __init__(self, config: SmoothingConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[Side, _Track] = {}
        self._latest: HandFrame | None = None
        self._interval_ema: float | None = None  # ns
        self._latency_ema: float = 0.0           # ns

    @property
    def render_delay_ns(self) -> int:
        if self.config.render_delay_ms is not None:
            return int(self.config.render_delay_ms * 1e6)
        auto = (self._interval_ema or 0.0) + self._latency_ema
        return int(np.clip(auto, 0, MAX_AUTO_DELAY_MS * 1e6))

    def push(self, frame: HandFrame) -> None:
        latest = self._latest
        if latest is not None and frame.source != latest.source:
            self.reset()  # a different stream: its rate, clock and poses have nothing to do with the old one
            latest = None
        if latest is not None and frame.t_ns <= latest.t_ns:
            return  # stale or duplicate
        self._measure(frame, latest)
        self._latest = frame
        if not self.config.enabled:
            return
        for side, pose in frame.hands.items():
            self._push_hand(self._tracks.setdefault(side, _Track()), frame.t_ns, pose)

    def _measure(self, frame: HandFrame, latest: HandFrame | None) -> None:
        if latest is not None:
            dt = frame.t_ns - latest.t_ns
            if dt < MAX_MEASURED_NS:
                e = self._interval_ema
                self._interval_ema = dt if e is None else e + EMA_ALPHA * (dt - e)
        # Arrival latency (capture -> push): renders must wait for it too, or there's nothing newer to interpolate
        # towards. Timestamps on a foreign clock (tests, replays with offsets) fall outside the window and are ignored.
        lat = now_ns() - frame.t_ns
        if 0 <= lat < MAX_MEASURED_NS:
            self._latency_ema += EMA_ALPHA * (lat - self._latency_ema)

    def _push_hand(self, tr: _Track, t_ns: int, pose: HandPose) -> None:
        cfg = self.config
        if tr.last_seen_ns is not None and t_ns - tr.last_seen_ns > BETAS_RESET_NS:
            tr.betas_seen.clear()
            tr.betas_frozen = None
        if tr.last is not None and t_ns - tr.last.t_ns > cfg.hold_ms * 1e6:
            tr.last = None  # hand was lost: start a fresh track instead of gliding in from the old pose
            tr.history.clear()
        tr.last_seen_ns = t_ns

        has_orient = pose.global_orient is not None
        rv = np.zeros((16, 3))
        rv[1:] = pose.hand_pose
        if has_orient:
            rv[0] = pose.global_orient
        rot = quat_from_rotvec(rv)
        transl = None if pose.transl is None else pose.transl.astype(np.float64)
        betas = self._betas(tr, pose.betas)

        prev = tr.last
        if prev is None:
            tr.rot_speed[:] = 0
            tr.transl_speed[:] = 0
        else:
            dt = (t_ns - prev.t_ns) * 1e-9
            base = prev.rot
            if not (has_orient and prev.has_orient):
                base = base.copy()
                base[0] = rot[0]  # orientation appeared or vanished: nothing to filter against
                tr.rot_speed[0] = 0
            delta = quat_log_rel(base, rot)  # (16, 3)
            tr.rot_speed += _alpha(cfg.d_cutoff, dt) * (delta / dt - tr.rot_speed)
            a = _alpha(cfg.min_cutoff + cfg.beta * np.linalg.norm(tr.rot_speed, axis=1), dt)
            rot = quat_step(base, delta * a[:, None])
            if transl is not None and prev.transl is not None:
                d = transl - prev.transl
                tr.transl_speed += _alpha(cfg.d_cutoff, dt) * (d / dt - tr.transl_speed)
                a = _alpha(cfg.min_cutoff + cfg.beta * TRANSL_SPEED_SCALE * np.linalg.norm(tr.transl_speed), dt)
                transl = prev.transl + a * d
            else:
                tr.transl_speed[:] = 0

        tr.last = _Filtered(t_ns, rot, has_orient, transl, betas, pose.frame, pose.confidence)
        tr.history.append(tr.last)

    def _betas(self, tr: _Track, betas: np.ndarray | None) -> np.ndarray | None:
        n = self.config.freeze_betas_after
        if betas is None or n <= 0:
            return betas
        if tr.betas_frozen is not None:
            return tr.betas_frozen
        tr.betas_seen.append(betas)
        med = np.median(np.stack(tr.betas_seen), axis=0).astype(np.float32)
        if len(tr.betas_seen) >= n:
            tr.betas_frozen = med
            tr.betas_seen.clear()
        return med

    def sample(self, t_ns: int) -> HandFrame | None:
        latest = self._latest
        if latest is None:
            return None
        hold_ns = self.config.hold_ms * 1e6
        if not self.config.enabled:
            hands = latest.hands if t_ns - latest.t_ns <= hold_ns else {}
            return HandFrame(latest.t_ns, hands, latest.source)

        rt = t_ns - self.render_delay_ns
        hands = {}
        for side, tr in self._tracks.items():
            h = tr.history
            if not h or rt < h[0].t_ns:
                continue  # empty, or the hand hadn't appeared yet at render time
            if rt >= h[-1].t_ns:
                if rt - h[-1].t_ns <= hold_ns:
                    hands[side] = _to_pose(h[-1])
                continue
            while len(h) > 2 and h[1].t_ns <= rt:  # renders move forward: drop samples we're past
                h.popleft()
            i = 0
            while h[i + 1].t_ns <= rt:  # bounded: rt < h[-1].t_ns
                i += 1
            a, b = h[i], h[i + 1]
            u = (rt - a.t_ns) / (b.t_ns - a.t_ns)
            hands[side] = _interp(a, b, u)
        return HandFrame(rt, hands, latest.source)


def _to_pose(f: _Filtered) -> HandPose:
    rv = rotvec_from_quat(f.rot)
    return HandPose(rv[1:], rv[0] if f.has_orient else None, f.transl, f.betas, f.frame, f.confidence)


def _interp(a: _Filtered, b: _Filtered, u: float) -> HandPose:
    near = a if u < 0.5 else b
    rot = slerp(a.rot, b.rot, u)
    rv = rotvec_from_quat(rot)
    orient = rv[0] if a.has_orient and b.has_orient else rotvec_from_quat(near.rot[0]) if near.has_orient else None
    if a.transl is not None and b.transl is not None:
        transl = a.transl + u * (b.transl - a.transl)
    else:
        transl = near.transl
    return HandPose(rv[1:], orient, transl, near.betas, near.frame, a.confidence + u * (b.confidence - a.confidence))
