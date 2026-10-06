#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "python-dotenv", "mcap"]
# ///
"""Report what's actually inside the delivery, without downloading the recordings.

- Bucket layout: top-level prefixes and any keys mentioning EMG / MANO / hand / mesh / wrist.
- Per session: manifest.json and metadata.json contents.
- Per MCAP: channels (topics), schemas, message counts and duration, read from the MCAP summary
  section via HTTP range requests (a few MB per file, not the whole recording).

Usage:
    uv run inspect_bucket.py                 # first MCAP of each session
    uv run inspect_bucket.py --all-mcaps     # every MCAP (91 files, still only range reads)
    uv run inspect_bucket.py > report.txt
"""

import argparse
import io
import json
import re
from collections import Counter

from mcap.reader import make_reader

from pull_sessions import DATE_FOLDERS, SESSIONS, list_objects, make_client

KEYWORDS = re.compile(r"emg|mano|hand|mesh|wrist|myo|joint|pose|keypoint|imu|glove", re.I)


class S3RangeReader(io.RawIOBase):
    """Seekable read-only file over an S3 object, fetching byte ranges on demand."""

    def __init__(self, client, bucket, key, size):
        self.client, self.bucket, self.key, self.size, self.pos = client, bucket, key, size, 0
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = self.client.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={self.pos}-{end}")["Body"].read()
        b[: len(data)] = data
        self.pos += len(data)
        self.bytes_fetched += len(data)
        return len(data)


def show_json(client, bucket, key, limit=4000):
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    try:
        text = json.dumps(json.loads(body), indent=2)
    except json.JSONDecodeError:
        text = body.decode(errors="replace")
    print(f"--- {key} ---")
    print(text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more chars)")


def inspect_mcap(client, bucket, obj):
    raw = S3RangeReader(client, bucket, obj["Key"], obj["Size"])
    stream = io.BufferedReader(raw, buffer_size=1024 * 1024)
    print(f"--- {obj['Key']} ({obj['Size'] / 1024**3:.1f} GB) ---")
    try:
        summary = make_reader(stream).get_summary()
    except Exception as e:  # noqa: BLE001 - report and move on
        print(f"  could not read MCAP summary: {type(e).__name__}: {e}")
        return
    if summary is None:
        print("  no summary section (file isn't indexed; Encord also requires one)")
        return

    stats = summary.statistics
    counts = stats.channel_message_counts if stats else {}
    if stats:
        dur = (stats.message_end_time - stats.message_start_time) / 1e9
        print(f"  {stats.message_count} messages, {len(summary.channels)} channels, {dur:.1f}s, "
              f"{len(summary.chunk_indexes)} chunks")
    for cid, ch in sorted(summary.channels.items(), key=lambda kv: kv[1].topic):
        schema = summary.schemas.get(ch.schema_id)
        schema_name = f"{schema.name} ({schema.encoding})" if schema else "no schema"
        flag = "  <==" if KEYWORDS.search(ch.topic) or (schema and KEYWORDS.search(schema.name)) else ""
        print(f"  {ch.topic:<45} {schema_name:<55} {counts.get(cid, '?'):>8} msgs{flag}")
        if ch.metadata:
            print(f"      channel metadata: {ch.metadata}")
    for m in summary.metadata_indexes:
        print(f"  metadata record: {m.name}")
    for a in summary.attachment_indexes:
        print(f"  attachment: {a.name} ({a.media_type}, {a.data_size} bytes)")
    print(f"  (read {raw.bytes_fetched / 1024**2:.1f} MB of the file)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-mcaps", action="store_true", help="inspect every MCAP, not just the first per session")
    args = parser.parse_args()

    client, bucket = make_client()

    print("\n=== Bucket layout ===")
    top = client.list_objects_v2(Bucket=bucket, Delimiter="/")
    for p in top.get("CommonPrefixes", []):
        print(f"  {p['Prefix']}")
    for o in top.get("Contents", []):
        print(f"  {o['Key']} ({o['Size']} bytes)")

    all_objs = list(list_objects(client, bucket, ""))
    exts = Counter(o["Key"].rsplit(".", 1)[-1].lower() if "." in o["Key"].rsplit("/", 1)[-1] else "(none)" for o in all_objs)
    print(f"\n{len(all_objs)} objects in the whole bucket, {sum(o['Size'] for o in all_objs) / 1024**3:.1f} GB")
    print("By extension:", dict(exts.most_common()))
    session_keys = [o for o in all_objs if any(s in o["Key"] for s in SESSIONS)]
    other = [o for o in all_objs if o not in session_keys]
    print(f"Outside the 6 known sessions: {len(other)} objects")
    for o in other[:50]:
        print(f"  {o['Size'] / 1024**2:>10.1f} MB  {o['Key']}")
    if len(other) > 50:
        print(f"  ... {len(other) - 50} more")
    hits = [o["Key"] for o in all_objs if KEYWORDS.search(o["Key"])]
    print(f"\nKeys mentioning emg/mano/hand/mesh/wrist/pose/...: {len(hits)}")
    for k in hits[:30]:
        print(f"  {k}")

    for sid, name in SESSIONS.items():
        print(f"\n=== {name} ({sid}) ===")
        objs = [o for f in DATE_FOLDERS for o in list_objects(client, bucket, f"{f}/{sid}/")]
        for o in objs:
            if o["Key"].endswith(("manifest.json", "_metadata.json")):
                show_json(client, bucket, o["Key"])
        mcaps = [o for o in objs if o["Key"].endswith(".mcap")]
        for o in mcaps if args.all_mcaps else mcaps[:1]:
            inspect_mcap(client, bucket, o)


if __name__ == "__main__":
    main()
