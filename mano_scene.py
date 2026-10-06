"""Build an Encord scene (ego video + MANO hand meshes + EMG) from a Mudra EMG episode.

Usage:
    uv run mano_scene.py                                   # list episodes
    uv run mano_scene.py 0aa8de95                          # uuid prefix ...
    uv run mano_scene.py sub-P001Fer_task-Glue_ep-005      # ... or episode id
        [--fps 15] [--start S] [--duration S] [--device auto|cuda|mps|cpu] [--emg-hz HZ] [--no-upload]

Downloads from R2 with Wrangler using your Cloudflare login (`npx wrangler login` once), and uploads with `gcloud`.
The episode MCAP is downloaded to data/<uuid>/<uuid>.mcap and split into one file per topic under
data/<uuid>/extracted/ (see split_mcap). The scene is built from those files into out/<episode_id>/. Every stage
is skipped if its output already exists; delete a stage's output to redo it. See 2026-10-06-mano-scene-design.md.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

BUCKET = "zander-eeg-data"
PREFIX = "09_21_26_POC/"
GCS_BUCKET = "long-horizon-wristband-demo-data"
HERE = Path(__file__).parent

VIDEO_TOPIC = "/ego/zed_head/side_by_side/image_compressed"
CAMERA_INFO_TOPIC = "/ego/zed_head/left/camera_info"
EMG_TOPICS = {side: f"/mudra/wristband_{side}/emg" for side in ("left", "right")}
VIDEO_FPS = 60
SIDES = ("left", "right")
# Files that stay local (not uploaded to GCS).
LOCAL_ONLY = ("mano.npz",)
# Every uploaded scene, for registering in Encord in one go (it skips scenes it already has).
UPLOAD_JSON = HERE / "encord_upload.json"


# ---------------------------------------------------------------------------------------------- download


def wrangler(*args, **kwargs):
    """Run a Wrangler command with the user's Cloudflare login."""
    cmd = ["wrangler"] if shutil.which("wrangler") else ["npx", "--yes", "wrangler"]
    env = {**os.environ, "WRANGLER_SEND_METRICS": "false"}
    r = subprocess.run([*cmd, *args], env=env, capture_output=True, **kwargs)
    if r.returncode:
        err = r.stderr.decode(errors="replace") if isinstance(r.stderr, bytes) else r.stderr
        sys.exit(f"wrangler {' '.join(args)} failed (run `npx wrangler login` if you aren't logged in):\n{err[-2000:]}")
    return r


def r2_key(key):
    return f"{BUCKET}/{PREFIX}{key}"


def load_manifest():
    out = wrangler("r2", "object", "get", r2_key("manifest.json"), "--remote", "--pipe").stdout
    return json.loads(out)["files"]


def list_episodes(episodes):
    print(f"{'uuid':<10} {'episode_id':<44} {'kind':<6} {'dur s':>7} {'MB':>7}")
    for e in sorted(episodes, key=lambda e: e["episode_id"]):
        print(f"{e['uuid'][:8]:<10} {e['episode_id']:<44} {e['kind']:<6} {float(e['duration_s']):>7.1f} "
              f"{e['mcap_bytes'] / 1024**2:>7,.0f}")
    print("\nRun: uv run mano_scene.py <uuid prefix or episode_id>")


def download(entry, dest: Path):
    if dest.exists():
        return
    print(f"[download] {entry['mcap_key']} ({entry['mcap_bytes'] / 1024**2:,.0f} MB)")
    tmp = dest.with_suffix(".part")
    wrangler("r2", "object", "get", r2_key(entry["mcap_key"]), "--remote", "--file", str(tmp))
    h = hashlib.sha256()
    with tmp.open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    if h.hexdigest() != entry["mcap_sha256"]:
        sys.exit(f"SHA-256 mismatch for {tmp}; delete it and re-run.")
    tmp.rename(dest)


# ---------------------------------------------------------------------------------------------- split

# Each topic is written to extracted/<topic path><ext>, e.g. /mudra/wristband_left/emg -> mudra/wristband_left/emg.csv
#   foxglove.CompressedVideo  -> .h265 / .h264 elementary stream + .frames.csv (log_time_ns, offset, bytes per frame)
#                                + .mp4 (same stream remuxed for players; needs ffmpeg)
#   sensor_msgs/CameraInfo    -> .json (first message; flags if it ever changes)
#   sensor_msgs/Imu           -> .csv
#   mudra.Sample (JSON)       -> .csv, one row per sample, v0..vN are the channel values
#   anything else             -> .jsonl, one message per line
# plus summary.json (time range, per-topic schema / count / file) and metadata.json (MCAP metadata records).


def topic_file(root: Path, topic: str, ext: str):
    path = root / (topic.strip("/") + ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def to_plain(obj):
    """Decoded ROS 2 / protobuf message -> JSON-serialisable structure."""
    if hasattr(obj, "DESCRIPTOR"):
        from google.protobuf.json_format import MessageToDict
        return MessageToDict(obj)
    if isinstance(obj, (bytes, bytearray)):
        return {"bytes_len": len(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if hasattr(obj, "__slots__"):
        return {k: to_plain(getattr(obj, k)) for k in obj.__slots__}
    return obj


def stamp_ns(header):
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def split_mcap(mcap_path: Path, extracted: Path):
    """Split every topic of the MCAP into its own file under extracted/."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
    from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

    if extracted.exists():
        return
    tmp = extracted.with_name(extracted.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    print(f"[split] {mcap_path.name} -> {extracted}")

    factories = [Ros2DecoderFactory(), ProtobufDecoderFactory()]
    files, writers, topics, decoders = {}, {}, {}, {}

    def open_file(path: Path, mode="w"):
        if path not in files:
            files[path] = path.open(mode, **({"newline": ""} if mode == "w" else {}))
        return files[path]

    def csv_writer(path: Path, header):
        if path not in writers:
            writers[path] = csv.writer(open_file(path))
            writers[path].writerow(header)
        return writers[path]

    with mcap_path.open("rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        meta = [{"name": m.name, **m.metadata} for m in reader.iter_metadata()]
        (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))

        for n, (schema, channel, message) in enumerate(reader.iter_messages(log_time_order=False), 1):
            topic, t = channel.topic, message.log_time
            schema_name = schema.name if schema else ""
            info = topics.setdefault(topic, {"schema": schema_name, "encoding": channel.message_encoding, "count": 0})
            info["count"] += 1
            if n % 50000 == 0:
                print(f"\r[split] {n:,} messages", end="", flush=True)

            if channel.message_encoding == "json":
                d = json.loads(message.data)
                if schema_name == "mudra.Sample":
                    path = topic_file(tmp, topic, ".csv")
                    keys = [k for k in d if k != "v"]
                    w = csv_writer(path, ["log_time_ns", *keys, *(f"v{k}" for k in range(len(d["v"])))])
                    w.writerow([t, *(d.get(k) for k in keys), *d["v"]])
                else:
                    path = topic_file(tmp, topic, ".jsonl")
                    open_file(path).write(json.dumps({"log_time_ns": t, **d}) + "\n")
                info["file"] = str(path.relative_to(tmp))
                continue

            if channel.id not in decoders:
                decoders[channel.id] = next(
                    (d for fac in factories if (d := fac.decoder_for(channel.message_encoding, schema))), None)
            if decoders[channel.id] is None:
                info["file"] = None  # no decoder for this encoding
                continue
            msg = decoders[channel.id](message.data)

            if schema_name == "foxglove.CompressedVideo":
                fmt = (msg.format or "h264").lower()
                path = topic_file(tmp, topic, f".{fmt}")
                fh = open_file(path, "wb")
                w = csv_writer(topic_file(tmp, topic, ".frames.csv"), ["log_time_ns", "offset", "bytes"])
                w.writerow([t, fh.tell(), len(msg.data)])
                fh.write(msg.data)  # Annex-B NAL units; concatenated they form a playable stream
                info.update(file=str(path.relative_to(tmp)), format=fmt)
            elif schema_name == "sensor_msgs/msg/CameraInfo":
                path = topic_file(tmp, topic, ".json")
                d = {"width": msg.width, "height": msg.height, "distortion_model": msg.distortion_model,
                     "d": list(msg.d), "k": list(msg.k), "r": list(msg.r), "p": list(msg.p)}
                if not path.exists():
                    path.write_text(json.dumps(d, indent=2))
                    info["file"], info["first"] = str(path.relative_to(tmp)), d
                elif d != info["first"]:
                    info["changed"] = info.get("changed", 0) + 1
            elif schema_name == "sensor_msgs/msg/Imu":
                path = topic_file(tmp, topic, ".csv")
                w = csv_writer(path, ["log_time_ns", "stamp_ns", "orient_x", "orient_y", "orient_z", "orient_w",
                                      "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"])
                o, g, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
                w.writerow([t, stamp_ns(msg.header), o.x, o.y, o.z, o.w, g.x, g.y, g.z, a.x, a.y, a.z])
                info["file"] = str(path.relative_to(tmp))
            else:
                path = topic_file(tmp, topic, ".jsonl")
                open_file(path).write(json.dumps({"log_time_ns": t, "msg": to_plain(msg)}) + "\n")
                info["file"] = str(path.relative_to(tmp))

    for fh in files.values():
        fh.close()
    for info in topics.values():
        info.pop("first", None)
    stats = summary.statistics
    (tmp / "summary.json").write_text(json.dumps({
        "mcap": mcap_path.name, "message_start_time": stats.message_start_time,
        "message_end_time": stats.message_end_time, "topics": dict(sorted(topics.items())),
    }, indent=2))
    tmp.rename(extracted)
    print(f"\r[split] {sum(i['count'] for i in topics.values()):,} messages, {len(topics)} topics")


def remux_videos(extracted: Path):
    """Wrap each extracted elementary video stream in an .mp4 (no re-encode) so it plays in QuickTime/VLC."""
    summary = json.loads((extracted / "summary.json").read_text())
    for topic, info in summary["topics"].items():
        if info["schema"] != "foxglove.CompressedVideo" or not info.get("file"):
            continue
        src = extracted / info["file"]
        mp4 = src.with_suffix(".mp4")
        if mp4.exists():
            continue
        if not shutil.which("ffmpeg"):
            print(f"[split] skipping {mp4.name}: install ffmpeg to get playable .mp4 files")
            return
        with src.with_name(src.stem + ".frames.csv").open() as fh:
            times = [int(r["log_time_ns"]) for r in csv.DictReader(fh)]
        fps = (len(times) - 1) / ((max(times) - min(times)) / 1e9) if len(times) > 1 else VIDEO_FPS
        fps = round(fps) if abs(fps - round(fps)) < 0.1 else fps  # 59.99 measured -> 60
        tmp = mp4.with_name(mp4.stem + ".part.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-r", f"{fps:g}", "-i", str(src), "-c", "copy",
               *(["-tag:v", "hvc1"] if info.get("format") == "h265" else []), str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(f"[split] {mp4.name} remux failed: {r.stderr.strip()[-500:]}")
            continue
        tmp.rename(mp4)
        print(f"[split] {mp4.relative_to(extracted)} ({fps:.2f} fps)")


def extracted_file(extracted: Path, topic: str):
    summary = json.loads((extracted / "summary.json").read_text())
    info = summary["topics"].get(topic)
    if not info or not info.get("file"):
        sys.exit(f"{topic} isn't in {extracted}")
    return extracted / info["file"], info


# ---------------------------------------------------------------------------------------------- extract


def episode_window(extracted: Path, start_s, duration_s):
    """Clip window [t0, t1] in log_time ns, relative to the first message of the episode."""
    summary = json.loads((extracted / "summary.json").read_text())
    start, end = summary["message_start_time"], summary["message_end_time"]
    t0 = start + int((start_s or 0) * 1e9)
    t1 = end if duration_s is None else t0 + int(duration_s * 1e9)
    return t0, min(t1, end)


def extract_frames(extracted: Path, out: Path, t0, t1, fps):
    """Decode the side-by-side video, keep every Nth frame in [t0, t1], save the left eye as JPEG."""
    import av

    frames_dir = out / "frames"
    if frames_dir.exists():
        return
    tmp = out / "frames.part"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    video_path, info = extracted_file(extracted, VIDEO_TOPIC)
    with video_path.with_name(video_path.stem + ".frames.csv").open() as fh:
        index = [(int(r["log_time_ns"]), int(r["offset"]), int(r["bytes"])) for r in csv.DictReader(fh)]

    step = max(1, round(VIDEO_FPS / fps))
    codec = av.CodecContext.create({"h265": "hevc"}.get(info["format"], info["format"]), "r")
    log_times, kept = [], 0

    def save(frame):
        nonlocal kept
        i = frame.pts if frame.pts is not None else len(log_times) - 1
        t = log_times[i]
        if t0 <= t <= t1 and i % step == 0:
            img = frame.to_image()  # PIL, RGB
            img.crop((0, 0, img.width // 2, img.height)).save(tmp / f"{t}.jpg", quality=90)
            kept += 1

    with video_path.open("rb") as video:
        # Decode from the start (inter-frames need their keyframe), stop once past the window.
        for t, offset, size in index:
            if t > t1 + 10**9:
                break
            video.seek(offset)
            packet = av.Packet(video.read(size))
            packet.pts = packet.dts = len(log_times)
            packet.time_base = Fraction(1, VIDEO_FPS)
            log_times.append(t)
            for frame in codec.decode(packet):
                save(frame)
            if len(log_times) % 600 == 0:
                print(f"\r[extract] decoded {len(log_times)} frames, kept {kept}", end="", flush=True)
    for frame in codec.decode(None):
        save(frame)
    print(f"\r[extract] decoded {len(log_times)} frames, kept {kept} at {VIDEO_FPS / step:g} fps")
    tmp.rename(frames_dir)


def extract_camera(extracted: Path, out: Path):
    path = out / "camera.json"
    if not path.exists():
        shutil.copyfile(extracted_file(extracted, CAMERA_INFO_TOPIC)[0], path)


def extract_emg(extracted: Path, out: Path, emg_hz):
    """EMG per band -> CSV over the span of the saved frames (the time-series tiles show nothing past the last
    image). With emg_hz, reduce to an RMS envelope (mean-removed) at that rate."""
    paths = {side: out / f"emg_{side}.csv" for side in SIDES}
    if all(p.exists() for p in paths.values()):
        return
    times = frame_times(out)
    t0, t1 = times[0], times[-1]

    for side in SIDES:
        with extracted_file(extracted, EMG_TOPICS[side])[0].open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
            cols = [header.index("log_time_ns"), *(i for i, h in enumerate(header) if h.startswith("v"))]
            rows = [[float(r[i]) for i in cols] for r in reader if t0 <= int(r[cols[0]]) <= t1]
        if not rows:
            sys.exit(f"No samples on {EMG_TOPICS[side]} in the clip window")
        data = np.array(rows, dtype=np.float64)
        t, v = data[:, 0].astype(np.int64), data[:, 1:]
        if emg_hz:
            v = v - v.mean(axis=0)
            bins = (t - t0) // int(1e9 / emg_hz)
            edges = np.flatnonzero(np.diff(bins)) + 1
            starts = np.concatenate([[0], edges])
            t = t0 + bins[starts] * int(1e9 / emg_hz)
            v = np.sqrt(np.add.reduceat(v**2, starts, axis=0) / np.diff(np.append(starts, len(bins)))[:, None])
        with paths[side].open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([TIME_COLUMN, *(f"v{k}" for k in range(v.shape[1]))])
            for ti, vi in zip(t, v):
                w.writerow([to_scene_time(int(ti), t0), *(f"{x:.6g}" for x in vi)])
        print(f"[extract] emg_{side}.csv: {len(t)} rows")


# ---------------------------------------------------------------------------------------------- hands


def pick_device(name):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_wilor(out: Path, device_name):
    """WiLoR on every frame; keep the most confident detection per side. Writes mano.npz."""
    path = out / "mano.npz"
    if path.exists():
        return
    import cv2
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    cam = json.loads((out / "camera.json").read_text())
    fx, fy, cx, cy = cam["k"][0], cam["k"][4], cam["k"][2], cam["k"][5]
    frames = sorted((out / "frames").glob("*.jpg"), key=lambda p: int(p.stem))
    w, h = cam["width"], cam["height"]

    device = pick_device(device_name)
    print(f"[hands] WiLoR on {len(frames)} frames, device {device}")
    # WiLoR's full-image camera uses focal_length / 256 * max(w, h); choose it so that equals the real fx.
    pipe = WiLorHandPose3dEstimationPipeline(
        device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32,
        focal_length=fx * 256 / max(w, h), verbose=False,
        wilor_pretrained_dir=str(Path.home() / ".cache" / "wilor-mini"),
    )
    # WiLoR feeds its ViT a width-sliced (non-contiguous) tensor, which MPS convolutions reject
    # ("view size is not compatible with input tensor's size and stride"); hand every conv a contiguous input.
    for module in pipe.wilor_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_pre_hook(lambda _, inputs: (inputs[0].contiguous(), *inputs[1:]))

    n = len(frames)
    res = {f"{side}_{k}": np.zeros(shape, np.float32) for side in SIDES for k, shape in {
        "global_orient": (n, 3), "hand_pose": (n, 45), "betas": (n, 10), "cam_t": (n, 3),
        "vertices": (n, 778, 3), "bbox": (n, 4), "score": (n,),
    }.items()}
    res["log_time_ns"] = np.array([int(p.stem) for p in frames], np.int64)
    res["faces"] = np.asarray(pipe.wilor_model.mano.faces, np.int32)

    import time
    start = time.monotonic()
    for i, frame_path in enumerate(frames):
        image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
        boxes = pipe.hand_detector(image, conf=0.3, verbose=False)[0].boxes
        best = {}  # is_right -> (conf, xyxy)
        for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            if conf > best.get(int(cls), (0,))[0]:
                best[int(cls)] = (float(conf), xyxy)
        if best:
            is_rights = sorted(best)
            preds = pipe.predict_with_bboxes(image, np.stack([best[r][1] for r in is_rights]), is_rights)
            for r, pred in zip(is_rights, preds):
                side, p = SIDES[r], pred["wilor_preds"]
                cam_t = p["pred_cam_t_full"][0].copy()
                # WiLoR assumes the principal point is the image centre; shift to the real one.
                cam_t[0] += (w / 2 - cx) * cam_t[2] / fx
                cam_t[1] += (h / 2 - cy) * cam_t[2] / fy
                res[f"{side}_global_orient"][i] = p["global_orient"].reshape(3)
                res[f"{side}_hand_pose"][i] = p["hand_pose"].reshape(45)
                res[f"{side}_betas"][i] = p["betas"].reshape(10)
                res[f"{side}_cam_t"][i] = cam_t
                res[f"{side}_vertices"][i] = p["pred_vertices"][0]
                res[f"{side}_bbox"][i] = best[r][1]
                res[f"{side}_score"][i] = best[r][0]
        if (i + 1) % 25 == 0 or i + 1 == n:
            rate = (i + 1) / (time.monotonic() - start)
            print(f"\r[hands] {i + 1}/{n}  {rate:.2f} frames/s  ETA {(n - i - 1) / rate / 60:.1f} min",
                  end="", flush=True)
    print()
    for side in SIDES:
        print(f"[hands] {side} hand found in {(res[f'{side}_score'] > 0).sum()}/{n} frames")
    np.savez_compressed(out / "mano.part.npz", **res)
    (out / "mano.part.npz").rename(path)


def write_meshes(out: Path):
    """One OBJ per detected hand per frame, in the left camera's frame (OpenCV axes, metres)."""
    meshes = out / "meshes"
    if meshes.exists():
        return
    tmp = out / "meshes.part"
    shutil.rmtree(tmp, ignore_errors=True)
    m = np.load(out / "mano.npz")
    for side in SIDES:
        (tmp / side).mkdir(parents=True)
        # WiLoR mirrors left hands by negating x, which flips triangle winding; flip it back.
        faces = m["faces"] if side == "right" else m["faces"][:, ::-1]
        face_lines = "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)
        for t, score, verts, cam_t in zip(m["log_time_ns"], m[f"{side}_score"], m[f"{side}_vertices"],
                                          m[f"{side}_cam_t"]):
            if score <= 0:
                continue
            v = verts + cam_t
            (tmp / side / f"{t}.obj").write_text(
                f"o {side}_hand\n" + "".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in v) + face_lines)
    (tmp / "empty.obj").write_text(EMPTY_OBJ)
    tmp.rename(meshes)
    print(f"[meshes] {sum(1 for _ in meshes.rglob('*.obj'))} OBJ files")


# ---------------------------------------------------------------------------------------------- scene

# Encord scene timestamps are integers on one axis shared by every stream (and the time column of the
# time-series CSVs), and the viewer requires the first one to be < 10. We use integer milliseconds since the
# first saved video frame.
TIME_COLUMN = "time"
# The viewer keeps showing a model until its next event, so a hand that disappears gets this degenerate mesh
# (an empty OBJ fails to load).
EMPTY_OBJ = "o empty\nv 0 0 0\nv 0 0 0\nv 0 0 0\nf 1 2 3\n"
OPENCV = {"x": "right", "y": "down", "z": "forward"}


def to_scene_time(log_time_ns: int, t0: int):
    return round((log_time_ns - t0) / 1e6)


def frame_times(out: Path):
    return sorted(int(p.stem) for p in (out / "frames").glob("*.jpg"))


def camera_intrinsics(cam):
    k, d = cam["k"], list(cam["d"])
    intrinsics = {"type": "simple", "fx": k[0], "fy": k[4], "ox": k[2], "oy": k[5]}
    model = cam["distortion_model"]
    if not any(d):
        intrinsics["model"] = {"type": "pinhole"}
    elif model == "plumb_bob" and len(d) >= 5:
        intrinsics["model"] = dict(zip(("type", "k1", "k2", "t1", "t2", "k3"), ["plumb_bob", *d[:5]]))
    elif model == "rational_polynomial" and len(d) >= 8:
        intrinsics["model"] = dict(zip(("type", "k1", "k2", "t1", "t2", "k3", "k4", "k5", "k6"),
                                       ["rational_polynomial", *d[:8]]))
    else:
        sys.exit(f"Unsupported distortion model {model!r} with {len(d)} coefficients")
    return intrinsics


def build_scene(out: Path, url_base: str):
    """Scene with config: camera + image stream, one model stream per hand, one time series per EMG band.

    Everything is in the left camera's frame: the camera has no frame of reference (so it sits at the root with
    identity pose), model streams are always drawn at the root, and both conventions are OpenCV's.
    """
    cam = json.loads((out / "camera.json").read_text())
    times = frame_times(out)
    t0 = times[0]
    content = {
        "camera": {"type": "camera_parameters", "events": [{
            "timestamp": 0, "widthPx": cam["width"], "heightPx": cam["height"],
            "intrinsics": camera_intrinsics(cam),
        }]},
        "ego": {"type": "image", "camera": "camera", "events": [
            {"timestamp": to_scene_time(t, t0), "uri": f"{url_base}/frames/{t}.jpg"} for t in times]},
    }
    for side in SIDES:
        # The viewer reports a model stream as "not found" at any time before its first event, so every stream
        # starts at the first frame: with empty.obj if the hand isn't detected there.
        events, shown = [], True
        for t in times:
            has_mesh = (out / "meshes" / side / f"{t}.obj").exists()
            if has_mesh:
                events.append({"timestamp": to_scene_time(t, t0), "uri": f"{url_base}/meshes/{side}/{t}.obj"})
            elif shown:
                events.append({"timestamp": to_scene_time(t, t0), "uri": f"{url_base}/meshes/empty.obj"})
            shown = has_mesh
        content[f"{side}_hand"] = {"type": "model", "events": events}
    for side in SIDES:
        content[f"emg_{side}"] = {"type": "time_series", "uri": f"{url_base}/emg_{side}.csv"}

    channels = {f"v{k}": {"label": f"ch{k}"} for k in range(3)}
    tiles = {
        "ego": {"type": "image", "streamName": "ego"},
        "3d": {"type": "3d", "hasSideView": False, "showCameraSwitcher": True},
        **{f"emg_{side}": {"type": "timeseries", "streamName": f"emg_{side}",
                           "timeseriesSettings": {"channels": channels}} for side in SIDES},
    }
    return {
        "worldConvention": OPENCV,
        "cameraConvention": OPENCV,
        "content": content,
        "layout": {
            "tiles": tiles,
            "layout": {
                "direction": "column", "splitPercentage": 65,
                "first": {"direction": "row", "first": "ego", "second": "3d"},
                "second": {"direction": "row", "first": "emg_left", "second": "emg_right"},
            },
        },
    }


# ---------------------------------------------------------------------------------------------- upload


def add_to_upload_json(item):
    """Add (or replace, by title) one scene in the repo-root encord_upload.json."""
    scenes = json.loads(UPLOAD_JSON.read_text())["scenes"] if UPLOAD_JSON.exists() else []
    scenes = sorted([s for s in scenes if s["title"] != item["title"]] + [item], key=lambda s: s["title"])
    tmp = UPLOAD_JSON.with_suffix(".part")
    tmp.write_text(json.dumps({"scenes": scenes}, indent=2))
    tmp.rename(UPLOAD_JSON)
    return len(scenes)


def upload(out: Path, uuid: str):
    dest = f"gs://{GCS_BUCKET}/{uuid}/"
    exclude = "|".join(f"^{name.replace('.', chr(92) + '.')}$" for name in LOCAL_ONLY)
    print(f"[upload] {out} -> {dest}")
    subprocess.run(["gcloud", "storage", "rsync", "--recursive", f"--exclude={exclude}", str(out), dest], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", nargs="?", help="uuid prefix or episode_id; omit to list episodes")
    parser.add_argument("--fps", type=float, default=15, help="frames sampled per second of video (default 15)")
    parser.add_argument("--start", type=float, help="clip start, seconds after episode start")
    parser.add_argument("--duration", type=float, help="clip length in seconds")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--emg-hz", type=float, default=100,
                        help="RMS envelope rate for the EMG plots (default 100; 0 = raw 2133 Hz samples, which "
                             "share millisecond timestamps)")
    parser.add_argument("--no-upload", action="store_true", help="stop after writing local files + JSON")
    args = parser.parse_args()

    episodes = load_manifest()
    if not args.episode:
        list_episodes(episodes)
        return
    matches = [e for e in episodes if e["uuid"].startswith(args.episode) or e["episode_id"] == args.episode]
    if len(matches) != 1:
        sys.exit(f"{len(matches)} episodes match {args.episode!r}; run without arguments to list them.")
    entry = matches[0]
    episode_id = entry["episode_id"]

    out = HERE / "out" / episode_id
    out.mkdir(parents=True, exist_ok=True)
    # Cached stages are only valid for the same clip settings.
    settings = {"fps": args.fps, "start": args.start, "duration": args.duration, "emg_hz": args.emg_hz}
    settings_path = out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        sys.exit(f"{out} was built with {settings_path.read_text().strip()}; delete it "
                 "or use the same options.")
    settings_path.write_text(json.dumps(settings))

    data = HERE / "data" / entry["uuid"]
    data.mkdir(parents=True, exist_ok=True)
    mcap_path = data / Path(entry["mcap_key"]).name
    download(entry, mcap_path)
    extracted = data / "extracted"
    split_mcap(mcap_path, extracted)
    remux_videos(extracted)
    t0, t1 = episode_window(extracted, args.start, args.duration)
    extract_camera(extracted, out)
    extract_frames(extracted, out, t0, t1, args.fps)
    extract_emg(extracted, out, args.emg_hz)
    run_wilor(out, args.device)
    write_meshes(out)

    # One bucket folder per episode, keyed by its R2 uuid (as in data/), holding every file the scene references.
    url_base = f"gs://{GCS_BUCKET}/{entry['uuid']}"
    scene = build_scene(out, url_base)
    (out / "scene.json").write_text(json.dumps(scene, indent=2))
    metadata = {k: entry[k] for k in ("uuid", "episode_id", "kind", "duration_s") if k in entry}
    metadata.update(settings)
    print(f"[scene] wrote {out / 'scene.json'}")

    if args.no_upload:
        print(f"Not uploaded, so not added to {UPLOAD_JSON.name}.")
        return
    upload(out, entry["uuid"])
    n = add_to_upload_json({"title": episode_id, "scene": scene, "clientMetadata": metadata})
    print(f"\nDone. {UPLOAD_JSON.name} now lists {n} scene(s); register it in Encord against the {GCS_BUCKET} "
          "integration.")


if __name__ == "__main__":
    main()
