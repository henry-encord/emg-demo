#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "python-dotenv", "mcap", "mcap-ros2-support", "mcap-protobuf-support"]
# ///
"""Download one episode MCAP from zander-eeg-data/09_21_26_POC/ and split it into per-stream files.

Usage:
    uv run emg.py                                  # list episodes
    uv run emg.py 5c5be84a                         # uuid prefix ...
    uv run emg.py sub-P001Fer_task-Glue_ep-005     # ... or episode id

Needs ACCESS_KEY_ID, SECRET_ACCESS_KEY and ENDPOINT_URL (or R2_ACCOUNT_ID) in .env.

Output, in out/<episode_id>/:
    mudra/wristband_{left,right}/{emg,imu_h,ppg}.csv   one row per sample; v0..vN are the channel values
    video.h265 (+ video.mp4 if ffmpeg is installed)     head camera, left|right side by side, 60 fps
    imu_head.csv, camera_info_{left,right}.json
    <topic>.jsonl                                       events, frame sync, band clocks (raw JSON messages)
    metadata.json                                       MCAP metadata records (episode, subject, calibration ids, ...)
"""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

BUCKET = "zander-eeg-data"
PREFIX = "09_21_26_POC/"
HERE = Path(__file__).parent


def make_client():
    load_dotenv(HERE.parent / ".env")
    endpoint = os.getenv("ENDPOINT_URL") or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3", endpoint_url=endpoint, region_name="auto",
        aws_access_key_id=os.environ["ACCESS_KEY_ID"], aws_secret_access_key=os.environ["SECRET_ACCESS_KEY"],
    )


def load_manifest(client):
    body = client.get_object(Bucket=BUCKET, Key=PREFIX + "manifest.json")["Body"].read()
    return json.loads(body)["files"]


def download(client, entry, dest: Path):
    if dest.exists() and dest.stat().st_size == entry["mcap_bytes"]:
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    total, done = entry["mcap_bytes"], 0

    def progress(n):
        nonlocal done
        done += n
        print(f"\r  {done / 1024**2:,.0f} / {total / 1024**2:,.0f} MB", end="", flush=True)

    print(f"Downloading {entry['mcap_key']}")
    client.download_file(BUCKET, PREFIX + entry["mcap_key"], str(tmp), Callback=progress)
    print()

    h = hashlib.sha256()
    with tmp.open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    if h.hexdigest() != entry["mcap_sha256"]:
        sys.exit(f"SHA-256 mismatch for {tmp}; delete it and re-run.")
    tmp.rename(dest)


def split(mcap_path: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files, writers = {}, {}

    def csv_writer(path: Path, header):
        if path not in writers:
            path.parent.mkdir(parents=True, exist_ok=True)
            files[path] = path.open("w", newline="")
            writers[path] = csv.writer(files[path])
            writers[path].writerow(header)
        return writers[path]

    def text_file(path: Path, mode="w"):
        if path not in files:
            files[path] = path.open(mode)
        return files[path]

    counts = {}
    with mcap_path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[Ros2DecoderFactory(), ProtobufDecoderFactory()])
        meta = [{"name": m.name, **m.metadata} for m in reader.iter_metadata()]
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        channels = reader.get_summary().channels.values()
        json_topics = [c.topic for c in channels if c.message_encoding == "json"]
        binary_topics = [c.topic for c in channels if c.message_encoding != "json"]

        # Pass 1: JSON topics (EMG / wrist IMU / PPG samples, events, sync, clocks).
        for schema, channel, message in reader.iter_messages(topics=json_topics):
            topic, t = channel.topic, message.log_time
            counts[topic] = counts.get(topic, 0) + 1
            if schema.name == "mudra.Sample":  # /mudra/wristband_<side>/<stream>
                s = json.loads(message.data)
                _, _, band, stream = topic.split("/")
                w = csv_writer(out / "mudra" / band / f"{stream}.csv",
                               ["log_time_ns", "t_host_ns", "t_mono_ns", "t_lsl_ns", "t_dev", "pkg_seq", "i"]
                               + [f"v{k}" for k in range(len(s["v"]))])
                w.writerow([t, s["t_host_ns"], s["t_mono_ns"], s.get("t_lsl_ns"), s["t_dev"], s["pkg_seq"], s["i"], *s["v"]])
            else:
                text_file(out / (topic.strip("/").replace("/", "__") + ".jsonl")).write(
                    json.dumps({"log_time_ns": t, **json.loads(message.data)}) + "\n")

        # Pass 2: ROS2 / protobuf topics (video, head IMU, camera info).
        video_fmt = None
        for schema, channel, message, msg in reader.iter_decoded_messages(topics=binary_topics):
            t = message.log_time
            counts[channel.topic] = counts.get(channel.topic, 0) + 1
            if schema.name == "foxglove.CompressedVideo":
                video_fmt = video_fmt or (msg.format or "h265").lower()
                text_file(out / f"video.{video_fmt}", "wb").write(msg.data)
            elif schema.name == "sensor_msgs/msg/Imu":
                w = csv_writer(out / "imu_head.csv", ["log_time_ns", "stamp_ns", "orient_x", "orient_y", "orient_z", "orient_w",
                                                      "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"])
                o, g, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
                w.writerow([t, msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
                            o.x, o.y, o.z, o.w, g.x, g.y, g.z, a.x, a.y, a.z])
            elif schema.name == "sensor_msgs/msg/CameraInfo":
                side = "left" if "/left/" in channel.topic else "right"
                path = out / f"camera_info_{side}.json"
                if not path.exists():
                    path.write_text(json.dumps({
                        "width": msg.width, "height": msg.height, "distortion_model": msg.distortion_model,
                        "d": list(msg.d), "k": list(msg.k), "r": list(msg.r), "p": list(msg.p),
                    }, indent=2))

    for fh in files.values():
        fh.close()

    if video_fmt and shutil.which("ffmpeg"):
        src = out / f"video.{video_fmt}"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "60", "-i", str(src),
                            "-c", "copy", str(out / "video.mp4")], capture_output=True, text=True)
        if r.returncode:
            print(f"mp4 remux failed: {r.stderr.strip()[:300]}")

    print(f"\nWrote {out.resolve()}")
    for topic, n in sorted(counts.items()):
        print(f"  {topic:<48} {n:>9,} messages")


def main():
    client = make_client()
    episodes = load_manifest(client)

    if len(sys.argv) < 2:
        print(f"{'uuid':<10} {'episode_id':<44} {'kind':<6} {'dur s':>7} {'MB':>7}")
        for e in sorted(episodes, key=lambda e: e["episode_id"]):
            print(f"{e['uuid'][:8]:<10} {e['episode_id']:<44} {e['kind']:<6} {float(e['duration_s']):>7.1f} "
                  f"{e['mcap_bytes'] / 1024**2:>7,.0f}")
        print("\nRun: uv run emg.py <uuid prefix or episode_id>")
        return

    query = sys.argv[1]
    matches = [e for e in episodes if e["uuid"].startswith(query) or e["episode_id"] == query]
    if len(matches) != 1:
        sys.exit(f"{len(matches)} episodes match {query!r}; run without arguments to list them.")
    entry = matches[0]

    mcap_path = HERE / "data" / f"{entry['episode_id']}.mcap"
    download(client, entry, mcap_path)
    split(mcap_path, HERE / "out" / entry["episode_id"])


if __name__ == "__main__":
    main()
