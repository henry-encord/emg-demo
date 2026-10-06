#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["mcap", "mcap-ros2-support", "mcap-protobuf-support"]
# ///
"""Split an MCAP into one output per stream, each in its native format, for inspection.

    foxglove.CompressedVideo     -> <topic>.h264 / .h265 raw stream (+ .mp4 if ffmpeg is installed) + frames.csv
    sensor_msgs/CompressedImage  -> one image file per message in the format it was stored in (png/jpg) + frames.csv
    sensor_msgs/CameraInfo       -> camera_info.json (calibration; flags if it ever changes)
    sensor_msgs/Imu              -> imu.csv
    nav_msgs/Odometry            -> odom.csv
    tf2_msgs/TFMessage           -> transforms.csv (/tf) or static_transforms.json (/tf_static)
    anything else                -> messages.jsonl
    MCAP metadata records        -> _mcap_metadata.json

Usage:
    uv run extract_streams.py data/.../clip_merged.mcap
    uv run extract_streams.py clip.mcap --out extracted --topics '/imu/.*' '/wrist_.*'
"""

import argparse
import base64
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def to_plain(obj):
    """Decoded ROS2 / protobuf message -> JSON-serialisable structure."""
    if hasattr(obj, "DESCRIPTOR"):  # protobuf
        from google.protobuf.json_format import MessageToDict
        return MessageToDict(obj)
    if isinstance(obj, (bytes, bytearray)):
        return {"bytes_b64": base64.b64encode(obj).decode()} if len(obj) <= 4096 else {"bytes_len": len(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if hasattr(obj, "__slots__"):
        return {k: to_plain(getattr(obj, k)) for k in obj.__slots__}
    return obj


def xyz(v):
    return [v.x, v.y, v.z]


def xyzw(q):
    return [q.x, q.y, q.z, q.w]


class Writer:
    """Base: one output directory per topic."""

    def __init__(self, root: Path, topic: str, schema: str):
        self.dir = root / (topic.strip("/").replace("/", "__") or "root")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.topic, self.schema, self.count = topic, schema, 0
        self.note = ""

    def write(self, log_time, msg):
        raise NotImplementedError

    def close(self):
        pass

    def csv(self, name, header):
        f = (self.dir / name).open("w", newline="")
        w = csv.writer(f)
        w.writerow(header)
        return f, w


class VideoWriter(Writer):
    def __init__(self, *a):
        super().__init__(*a)
        self.stream = None
        self.fmt = None
        self.idx_f, self.idx = self.csv("frames.csv", ["frame", "log_time_ns", "stamp_ns", "bytes", "frame_id"])

    def write(self, log_time, msg):
        if self.stream is None:
            self.fmt = (msg.format or "h264").lower()
            self.path = self.dir / f"{self.dir.name}.{self.fmt}"
            self.stream = self.path.open("wb")
        self.stream.write(msg.data)  # Annex-B NAL units, concatenated = playable elementary stream
        ts = msg.timestamp.seconds * 1_000_000_000 + msg.timestamp.nanos
        self.idx.writerow([self.count, log_time, ts, len(msg.data), msg.frame_id])
        self.count += 1

    def close(self):
        self.idx_f.close()
        if not self.stream:
            return
        self.stream.close()
        self.note = f"{self.path.name} (raw {self.fmt})"
        if shutil.which("ffmpeg") and self.fmt in ("h264", "h265"):
            mp4 = self.path.with_suffix(".mp4")
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(self.path), "-c", "copy", str(mp4)],
                capture_output=True, text=True,
            )
            self.note += f" + {mp4.name}" if r.returncode == 0 else f" (mp4 remux failed: {r.stderr.strip()[:200]})"
        else:
            self.note += " — install ffmpeg to also get an .mp4"


class ImageWriter(Writer):
    """sensor_msgs/CompressedImage: keep each frame's bytes as-is (stripping only the compressedDepth header)."""

    def __init__(self, *a):
        super().__init__(*a)
        self.frames = self.dir / "frames"
        self.frames.mkdir(exist_ok=True)
        self.idx_f, self.idx = self.csv("frames.csv", ["frame", "file", "log_time_ns", "stamp_ns", "format", "bytes"])
        self.depth_header = None

    def write(self, log_time, msg):
        data, fmt = bytes(msg.data), msg.format
        if "compresseddepth" in fmt.lower() and (i := data.find(PNG_MAGIC)) > 0:
            # compressed_depth_image_transport prefixes a 12-byte header: int32 format, float32 depthQuantA, depthQuantB
            if self.depth_header is None:
                import struct
                self.depth_header = dict(zip(("format_enum", "depthQuantA", "depthQuantB"), struct.unpack("<iff", data[:12])))
            data = data[i:]
        ext = ".png" if data.startswith(PNG_MAGIC) else ".jpg" if data[:2] == b"\xff\xd8" else ".bin"
        name = f"{self.count:06d}_{log_time}{ext}"
        (self.frames / name).write_bytes(data)
        self.idx.writerow([self.count, name, log_time, stamp_ns(msg.header), fmt, len(data)])
        if self.count == 0:
            self.note = f"frames/*{ext} (format field: {fmt!r})"
        self.count += 1

    def close(self):
        self.idx_f.close()
        if self.depth_header:
            (self.dir / "compressed_depth_header.json").write_text(json.dumps(self.depth_header, indent=2))
            q = self.depth_header
            self.note += ("; 16-bit PNG values are depth in mm" if q["depthQuantA"] == 0 and q["depthQuantB"] == 0
                          else "; quantised 32FC1 depth: metres = depthQuantA / (value - depthQuantB)")


class CameraInfoWriter(Writer):
    def __init__(self, *a):
        super().__init__(*a)
        self.first = None
        self.changed = 0

    def write(self, log_time, msg):
        d = to_plain(msg)
        d.pop("header", None)
        if self.first is None:
            self.first = d
        elif d != self.first:
            self.changed += 1
        self.count += 1

    def close(self):
        (self.dir / "camera_info.json").write_text(json.dumps(self.first, indent=2))
        self.note = "camera_info.json" + (f" (WARNING: changed in {self.changed} messages)" if self.changed else " (constant)")


class ImuWriter(Writer):
    def __init__(self, *a):
        super().__init__(*a)
        self.f, self.w = self.csv("imu.csv", [
            "log_time_ns", "stamp_ns", "frame_id",
            "orient_x", "orient_y", "orient_z", "orient_w",
            "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z",
        ])
        self.note = "imu.csv (gyro rad/s, accel m/s^2)"

    def write(self, log_time, m):
        self.w.writerow([log_time, stamp_ns(m.header), m.header.frame_id,
                         *xyzw(m.orientation), *xyz(m.angular_velocity), *xyz(m.linear_acceleration)])
        self.count += 1

    def close(self):
        self.f.close()


class OdomWriter(Writer):
    def __init__(self, *a):
        super().__init__(*a)
        self.f, self.w = self.csv("odom.csv", [
            "log_time_ns", "stamp_ns", "frame_id", "child_frame_id",
            "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w",
            "vel_x", "vel_y", "vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z",
        ])
        self.note = "odom.csv"

    def write(self, log_time, m):
        p, t = m.pose.pose, m.twist.twist
        self.w.writerow([log_time, stamp_ns(m.header), m.header.frame_id, m.child_frame_id,
                         *xyz(p.position), *xyzw(p.orientation), *xyz(t.linear), *xyz(t.angular)])
        self.count += 1

    def close(self):
        self.f.close()


class TfWriter(Writer):
    def __init__(self, *a):
        super().__init__(*a)
        self.static = self.topic.endswith("_static")
        self.latest = {}
        if not self.static:
            self.f, self.w = self.csv("transforms.csv", [
                "log_time_ns", "stamp_ns", "parent", "child", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

    def write(self, log_time, m):
        for t in m.transforms:
            row = [*xyz(t.transform.translation), *xyzw(t.transform.rotation)]
            if self.static:
                self.latest[(t.header.frame_id, t.child_frame_id)] = row
            else:
                self.w.writerow([log_time, stamp_ns(t.header), t.header.frame_id, t.child_frame_id, *row])
        self.count += 1

    def close(self):
        if self.static:
            out = [{"parent": p, "child": c, "translation": r[:3], "rotation_xyzw": r[3:]} for (p, c), r in self.latest.items()]
            (self.dir / "static_transforms.json").write_text(json.dumps(out, indent=2))
            self.note = f"static_transforms.json ({len(out)} unique frame pairs)"
        else:
            self.f.close()
            self.note = "transforms.csv"


class JsonlWriter(Writer):
    def __init__(self, *a):
        super().__init__(*a)
        self.f = (self.dir / "messages.jsonl").open("w")
        self.note = "messages.jsonl"

    def write(self, log_time, m):
        self.f.write(json.dumps({"log_time_ns": log_time, "msg": to_plain(m)}) + "\n")
        self.count += 1

    def close(self):
        self.f.close()


WRITERS = {
    "foxglove.CompressedVideo": VideoWriter,
    "sensor_msgs/msg/CompressedImage": ImageWriter,
    "sensor_msgs/msg/CameraInfo": CameraInfoWriter,
    "sensor_msgs/msg/Imu": ImuWriter,
    "nav_msgs/msg/Odometry": OdomWriter,
    "tf2_msgs/msg/TFMessage": TfWriter,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", type=Path)
    parser.add_argument("--out", type=Path, help="output dir (default: <mcap name>_streams next to the file)")
    parser.add_argument("--topics", nargs="+", metavar="REGEX", help="only topics matching any of these regexes")
    args = parser.parse_args()

    out = args.out or args.mcap.with_name(args.mcap.stem + "_streams")
    out.mkdir(parents=True, exist_ok=True)
    wanted = [re.compile(p) for p in args.topics] if args.topics else None

    writers: dict[int, Writer] = {}
    with args.mcap.open("rb") as f:
        reader = make_reader(f, decoder_factories=[Ros2DecoderFactory(), ProtobufDecoderFactory()])
        summary = reader.get_summary()
        topics = None
        if wanted and summary:
            topics = [c.topic for c in summary.channels.values() if any(p.search(c.topic) for p in wanted)]
            print(f"Extracting {len(topics)} topics: {', '.join(sorted(topics))}")

        meta = [{"name": m.name, **m.metadata} for m in reader.iter_metadata()]  # names repeat (one per camera)
        (out / "_mcap_metadata.json").write_text(json.dumps(meta, indent=2))

        total = summary.statistics.message_count if summary and summary.statistics else None
        n = 0
        for schema, channel, message, decoded in reader.iter_decoded_messages(topics=topics, log_time_order=False):
            w = writers.get(channel.id)
            if w is None:
                cls = WRITERS.get(schema.name if schema else "", JsonlWriter)
                w = writers[channel.id] = cls(out, channel.topic, schema.name if schema else "")
            w.write(message.log_time, decoded)
            n += 1
            if n % 20000 == 0:
                print(f"  {n}{f' / {total}' if total else ''} messages", flush=True)

    print(f"\nWrote {out.resolve()}")
    summary_rows = []
    for w in sorted(writers.values(), key=lambda w: w.topic):
        w.close()
        summary_rows.append({"topic": w.topic, "schema": w.schema, "messages": w.count,
                             "dir": w.dir.name, "output": w.note})
        print(f"  {w.topic:<42} {w.count:>7}  -> {w.dir.name}/{w.note}")
    (out / "_streams.json").write_text(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
