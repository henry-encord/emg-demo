"""SyntheticSource: fingers curling and uncurling on sines. A stand-in for an EMG source: it emits only `finger_pose`
(no wrist orientation, position or shape), at a steady rate, stamped with the monotonic clock like a real source.

Flexion axes come from the SOMA rest pose (right hand): fingers point along -x from the wrist, the palm faces -z and
the thumb sits on the -y side. Finger joints flex about -y, which swings the tips towards -z (into the palm); the
metacarpals (…1) stay still, as they mostly do. The thumb flexes about mostly -z with a little +x, which bends it
across the palm towards the little finger. tests/test_synthetic.py checks both against the SOMA mesh.

Both sides use the same numbers: SOMA's left hand mirrors the right for the same pose (see core/types.py).
"""

from __future__ import annotations

import threading

import numpy as np

from hand_viewer.core.types import FINGER_JOINTS, SIDES, HandFrame, HandPose, PoseSink, now_ns

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_AXIS = np.array([0.0, -1.0, 0.0])
_THUMB_AXIS = np.array([0.3, 0.0, -1.0]) / np.linalg.norm([0.3, 0.0, -1.0])
# Peak flexion (rad) per finger_pose row. Thumb: CMC, MCP, IP, end. Others: metacarpal, MCP, PIP, DIP, end.
_PEAK = {"Thumb": (0.4, 0.6, 0.9, 0.0), "other": (0.0, 1.3, 1.5, 1.0, 0.0)}
AXES = np.array([_THUMB_AXIS if j.startswith("Thumb") else _FINGER_AXIS for j in FINGER_JOINTS])  # (24, 3)
MAX_FLEX = np.concatenate([_PEAK["Thumb"]] + [_PEAK["other"]] * 4)                                 # (24,)
FINGER_OF_ROW = np.array([FINGERS.index(next(f for f in FINGERS if j.lower().startswith(f))) for j in FINGER_JOINTS])
PHASE = {"index": 0.0, "middle": 0.6, "ring": 1.2, "pinky": 1.8, "thumb": 2.6}  # rad: a wave across the hand
PERIOD_S = 3.0


def curl_pose(curl: np.ndarray) -> np.ndarray:
    """SOMA finger_pose (24, 3) for per-finger curl amounts in [0, 1] (FINGERS order). Same for either side."""
    per_joint = np.asarray(curl, dtype=np.float64)[FINGER_OF_ROW] * MAX_FLEX
    return AXES * per_joint[:, None]


class SyntheticSource:
    name = "synthetic"

    def __init__(self, rate_hz: float = 60.0, sides=SIDES):
        self.rate_hz = rate_hz
        self.sides = tuple(sides)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, sink: PoseSink) -> None:
        if self._thread is not None:
            raise RuntimeError("SyntheticSource can only be started once")
        self._thread = threading.Thread(target=self._run, args=(sink,), name="synthetic-pose", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():  # stop() from inside a sink callback can't join
            t.join()

    def frame_at(self, t_ns: int) -> HandFrame:
        t = t_ns * 1e-9
        hands = {}
        for k, side in enumerate(self.sides):
            phase = np.array([PHASE[f] for f in FINGERS]) + k * np.pi / 2  # hands out of step, to tell them apart
            curl = 0.5 - 0.5 * np.cos(2 * np.pi * t / PERIOD_S + phase)
            hands[side] = HandPose(curl_pose(curl))
        return HandFrame(t_ns, hands, self.name)

    def _run(self, sink: PoseSink) -> None:
        try:
            period_ns = int(1e9 / self.rate_hz)
            sink.status(f"synthetic hands at {self.rate_hz:g} Hz")
            next_ns = now_ns()
            while not self._stop.is_set():
                sink.pose(self.frame_at(now_ns()))
                next_ns += period_ns
                wait = next_ns - now_ns()
                if wait < -period_ns:  # fell behind (e.g. machine asleep): don't burst to catch up
                    next_ns = now_ns()
                elif wait > 0 and self._stop.wait(wait * 1e-9):
                    break
        except Exception as e:  # noqa: BLE001 - the lifecycle says errors go to the sink, never out of the worker
            sink.error(f"synthetic source failed: {e!r}")
