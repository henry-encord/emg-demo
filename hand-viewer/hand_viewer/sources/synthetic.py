"""SyntheticSource: fingers curling and uncurling on sines. A stand-in for an EMG source: it emits only `hand_pose`
(no orientation, position or shape), at a steady rate, stamped with the monotonic clock like a real source.

Flexion axes come from the MANO rest pose (flat_hand_mean, right hand), where joint frames are aligned with the
model frame: fingers point along -x from the wrist, the thumb lies along +z, and the palm faces -y. Finger joints
(index, middle, pinky, ring; MANO's hand_pose order) flex about +z, which swings the tips towards -y. The thumb
flexes about a mix of +x and -y, which bends it into the palm and across towards the little finger. Checked by
running smplx's MANO (tests/test_synthetic.py does it again).

Left hands use WiLoR's mirrored convention (see core/types.py): the right-hand rows with y and z negated.
"""

from __future__ import annotations

import threading

import numpy as np

from hand_viewer.core.types import SIDES, HandFrame, HandPose, PoseSink, Side, now_ns

FINGERS = ("index", "middle", "pinky", "ring", "thumb")  # MANO hand_pose order, 3 joints each (base -> tip)
_THUMB_AXIS = np.array([0.8, -0.6, 0.0])
AXES = np.array([[0.0, 0.0, 1.0]] * 12 + [_THUMB_AXIS] * 3)  # (15, 3) unit flexion axes, right hand
# Peak flexion (rad) per joint: MCP, PIP, DIP for fingers; CMC, MCP, IP for the thumb.
MAX_FLEX = np.array([1.3, 1.5, 1.0] * 4 + [0.4, 0.6, 0.9])
PHASE = {"index": 0.0, "middle": 0.6, "ring": 1.2, "pinky": 1.8, "thumb": 2.6}  # rad: a wave across the hand
PERIOD_S = 3.0
MIRROR = np.array([1.0, -1.0, -1.0])


def curl_pose(curl: np.ndarray, side: Side = "right") -> np.ndarray:
    """MANO hand_pose (15, 3) for per-finger curl amounts in [0, 1] (FINGERS order)."""
    per_joint = np.repeat(np.asarray(curl, dtype=np.float64), 3) * MAX_FLEX
    pose = AXES * per_joint[:, None]
    return pose * MIRROR if side == "left" else pose


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
            hands[side] = HandPose(curl_pose(curl, side))
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
