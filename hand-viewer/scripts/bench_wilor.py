"""Benchmark WilorModel (crop-then-blur, batched) against wilor_mini's predict_with_bboxes on recorded frames.

    MPLCONFIGDIR=$TMPDIR/mpl YOLO_CONFIG_DIR=$TMPDIR/yolo .venv/bin/python scripts/bench_wilor.py [--device mps]

Reports detect / pose / total ms and fps per (input long side, precision, path), plus how far fp16 and the library
path are from our fp32 output (max vertex and cam_t difference, mm).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_viewer.core.types import CameraIntrinsics, VideoFrame  # noqa: E402
from hand_viewer.sources.wilor import WilorModel, assign_sides, prepare  # noqa: E402

EPISODE = Path(__file__).resolve().parents[2] / "encord-scene/out/sub-P001Fer_task-Glue_ep-005"


def load_frames(episode: Path, start: int, stop: int):
    cam = json.loads((episode / "camera.json").read_text())
    k = CameraIntrinsics(cam["width"], cam["height"], cam["k"][0], cam["k"][4], cam["k"][2], cam["k"][5])
    paths = sorted((episode / "frames").glob("*.jpg"), key=lambda p: int(p.stem))[start:stop]
    return [VideoFrame(int(p.stem), cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), k) for p in paths]


def sync(model: WilorModel):
    import torch

    if model.device.type == "mps":
        torch.mps.synchronize()
    elif model.device.type == "cuda":
        torch.cuda.synchronize()


def run(model: WilorModel, frames, long_side: int, library: bool, warmup: int = 3):
    """Returns (mean detect ms, mean pose ms, per-frame results {side: (vertices, cam_t)})."""
    det_ms, pose_ms, outs = [], [], []
    for i, frame in enumerate(frames[:warmup] + frames):
        t0 = time.perf_counter()
        image, k = prepare(frame, long_side)
        hands, _ = assign_sides(model.detect(image), {})
        sync(model)
        t1 = time.perf_counter()
        if library:
            sides = list(hands)
            model.set_focal(k.fx, image.shape[1], image.shape[0])
            preds = model.pipe.predict_with_bboxes(image, np.stack([hands[s].box for s in sides]),
                                                   [int(s == "right") for s in sides]) if sides else []
            res = {s: (p["wilor_preds"]["pred_vertices"][0],
                       shift_principal(p["wilor_preds"]["pred_cam_t_full"][0], k)) for s, p in zip(sides, preds)}
        else:
            res = {r.side: (r.vertices, r.cam_t) for r in model.estimate(image, hands, k)}
        sync(model)
        t2 = time.perf_counter()
        if i >= warmup:
            det_ms.append((t1 - t0) * 1e3)
            pose_ms.append((t2 - t1) * 1e3)
            outs.append(res)
    return float(np.mean(det_ms)), float(np.mean(pose_ms)), outs


def shift_principal(cam_t, k: CameraIntrinsics):
    """The library assumes the principal point is the image centre; apply the same shift WilorModel does."""
    t = cam_t.copy()
    t[0] += (k.width / 2 - k.cx) * t[2] / k.fx
    t[1] += (k.height / 2 - k.cy) * t[2] / k.fy
    return t


def max_diff_mm(a, b, idx):
    d = [np.abs(x[s][idx] - y[s][idx]).max() for x, y in zip(a, b) for s in x if s in y]
    return 1e3 * max(d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--episode", type=Path, default=EPISODE)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--stop", type=int, default=140)
    ap.add_argument("--sides", type=int, nargs="+", default=[1920, 960])
    args = ap.parse_args()

    frames = load_frames(args.episode, args.start, args.stop)
    models = {False: WilorModel(args.device, fp16=False), True: WilorModel(args.device, fp16=True)}
    print(f"device {models[False].device}, {len(frames)} frames ({args.start}..{args.stop})")
    print(f"{'input':>6} {'prec':>5} {'path':>8} {'detect':>7} {'pose':>7} {'total':>7} {'fps':>5}  "
          f"{'dvert':>7} {'dcam_t':>7}")
    for long_side in args.sides:
        _, _, ref = run(models[False], frames, long_side, library=False, warmup=0)
        for fp16 in (False, True):
            for library in (True, False):
                d, p, outs = run(models[fp16], frames, long_side, library)
                print(f"{long_side:>6} {'fp16' if fp16 else 'fp32':>5} {'library' if library else 'ours':>8} "
                      f"{d:7.1f} {p:7.1f} {d + p:7.1f} {1e3 / (d + p):5.1f}  "
                      f"{max_diff_mm(outs, ref, 0):7.3f} {max_diff_mm(outs, ref, 1):7.2f}")
    print("dvert / dcam_t: max abs difference (mm) from our fp32 path at the same input size")


if __name__ == "__main__":
    main()
