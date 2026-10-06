#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "python-dotenv"]
# ///
"""Download individual clips (the .mcap plus its sidecar .json) and verify them against manifest.json.

Usage:
    uv run download_clip.py 6634df94                 # clip ID or prefix, as shown in the Encord title
    uv run download_clip.py 6634df94 b5c1abf2        # several clips
    uv run download_clip.py 6634df94 --out ~/clips
"""

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
from pathlib import Path

from boto3.s3.transfer import TransferConfig

from pull_sessions import DATE_FOLDERS, list_objects, make_client

# One big file at a time, so parallelise within the file.
TRANSFER_CONFIG = TransferConfig(multipart_chunksize=64 * 1024**2, max_concurrency=16)


class Progress:
    def __init__(self, total):
        self.total, self.done, self.start, self.lock = total, 0, time.monotonic(), threading.Lock()
        self.last_print = 0.0

    def __call__(self, n):
        with self.lock:
            self.done += n
            now = time.monotonic()
            if now - self.last_print < 1 and self.done < self.total:
                return
            self.last_print = now
            rate = self.done / max(now - self.start, 1e-6)
            eta = (self.total - self.done) / rate if rate else 0
            print(f"\r  {self.done / 1024**3:6.2f} / {self.total / 1024**3:.2f} GB  "
                  f"{rate / 1024**2:6.1f} MB/s  ETA {eta / 60:4.1f} min", end="", flush=True)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+", metavar="CLIP_ID", help="clip ID or prefix, e.g. 6634df94")
    parser.add_argument("--out", default="data", help="output directory (default: ./data)")
    parser.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    args = parser.parse_args()

    client, bucket = make_client()

    # Clip folders look like <date>/<session>/raw/<date>/<session>/<HH_MM_SS>/<clip_id>/<files>
    objs = [
        o for folder in DATE_FOLDERS for o in list_objects(client, bucket, f"{folder}/")
        if "/raw/" in o["Key"] and o["Key"].split("/")[-2].startswith(tuple(args.clips))
    ]
    found = {c for c in args.clips if any(o["Key"].split("/")[-2].startswith(c) for o in objs)}
    if missing := set(args.clips) - found:
        sys.exit(f"No clip found for: {', '.join(sorted(missing))} (searched {', '.join(DATE_FOLDERS)})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(o["Size"] for o in objs)
    free = shutil.disk_usage(out_dir).free
    print(f"{len(objs)} files, {total / 1024**3:.2f} GB; free space at {out_dir.resolve()}: {free / 1024**3:.1f} GB")
    if total > free:
        sys.exit("Not enough disk space; use --out on a bigger drive.")

    downloaded = []
    for o in sorted(objs, key=lambda o: o["Size"]):
        dest = out_dir / o["Key"]
        print(f"{o['Key'].split('/')[-1]}")
        if dest.exists() and dest.stat().st_size == o["Size"]:
            print("  already downloaded")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            client.download_file(bucket, o["Key"], str(tmp), Config=TRANSFER_CONFIG, Callback=Progress(o["Size"]))
            tmp.rename(dest)
            print()
        downloaded.append((o["Key"], dest))

    if args.no_verify:
        return

    # manifest.json lists sha256 per artifact, with paths relative to the session folder.
    print("\nVerifying against manifest.json")
    bad = 0
    manifests = {}
    for key, dest in downloaded:
        session_prefix = "/".join(key.split("/")[:2])
        if session_prefix not in manifests:
            body = client.get_object(Bucket=bucket, Key=f"{session_prefix}/manifest.json")["Body"].read()
            manifests[session_prefix] = {
                a["path"]: a["sha256"]
                for s in json.loads(body)["sessions"] for c in s["clips"] for a in c["artifacts"]
            }
        expected = manifests[session_prefix].get(key.removeprefix(session_prefix + "/"))
        if not expected:
            print(f"  not in manifest  {dest.name}")
        elif sha256(dest) == expected:
            print(f"  ok               {dest.name}")
        else:
            print(f"  MISMATCH         {dest.name}")
            bad += 1
    if bad:
        sys.exit(f"{bad} file(s) failed verification; delete them and re-run.")
    print(f"\nDone. Files in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
