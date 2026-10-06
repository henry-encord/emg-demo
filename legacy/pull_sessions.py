#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "python-dotenv"]
# ///
"""Download delivered sessions from a Cloudflare R2 (S3-compatible) bucket.

Usage:
    uv run pull_sessions.py --list          # show what's in the bucket, download nothing
    uv run pull_sessions.py                 # download the sessions below into ./data
    uv run pull_sessions.py --all           # download everything under the date folders
"""

import argparse
import hashlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from dotenv import load_dotenv

DATE_FOLDERS = ["01_10_2026", "02_10_2026"]

SESSIONS = {
    "6abe2c8699ef210001dfc12a": "bakery",
    "6abe432899ef210001dfc464": "grocery store",
    "6abf36ab4cff0b00018dd824": "carpet washing",
    "6abf73504cff0b00018dddcd": "appliance repair",
    "6abf8e0d4cff0b00018de0f8": "phone repair",
    "6abf96f44cff0b00018de228": "car repair",
}


TRANSFER_CONFIG = TransferConfig(multipart_chunksize=64 * 1024**2, max_concurrency=4)
CLIENT_CONFIG = Config(max_pool_connections=32, retries={"max_attempts": 10, "mode": "adaptive"})


def make_client():
    load_dotenv(Path(__file__).parent.parent / ".env")
    key_id = os.getenv("ACCESS_KEY_ID", "").strip()
    secret = os.getenv("SECRET_ACCESS_KEY", "").strip()
    if not key_id or not secret:
        sys.exit("Set ACCESS_KEY_ID and SECRET_ACCESS_KEY in .env")
    creds = {"aws_access_key_id": key_id, "aws_secret_access_key": secret}

    # AWS keys look like AKIA.../ASIA... (20 chars); R2 keys are 32 hex chars.
    is_aws = key_id.startswith(("AKIA", "ASIA"))
    endpoint = os.getenv("ENDPOINT_URL") or (
        f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com" if os.getenv("R2_ACCOUNT_ID") else None
    )
    if not endpoint and not is_aws:
        sys.exit(
            "This looks like a Cloudflare R2 key (not AWS). R2 needs the account ID, which can't be\n"
            "derived from the key — ask the sender for the endpoint URL\n"
            "(https://<account_id>.r2.cloudflarestorage.com) and the bucket name, then set\n"
            "ENDPOINT_URL (or R2_ACCOUNT_ID) and BUCKET in .env."
        )
    region = "auto" if endpoint else "us-east-1"
    client = boto3.client("s3", endpoint_url=endpoint, region_name=region, config=CLIENT_CONFIG, **creds)
    print(f"Provider: {'S3-compatible endpoint ' + endpoint if endpoint else 'AWS S3'}")

    bucket = os.getenv("BUCKET")
    if not bucket:
        try:
            buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
        except client.exceptions.ClientError as e:
            sys.exit(f"BUCKET not set and the key isn't allowed to list buckets ({e}). Ask the sender for the bucket name.")
        print(f"{len(buckets)} buckets visible to this key; looking for {', '.join(DATE_FOLDERS)} ...")
        matches = []
        for b in buckets:
            try:
                hits = [f for f in DATE_FOLDERS if client.list_objects_v2(Bucket=b, Prefix=f"{f}/", MaxKeys=1).get("KeyCount")]
            except client.exceptions.ClientError as e:
                print(f"  {b}: no access ({e.response['Error']['Code']})")
                continue
            if hits:
                print(f"  {b}: has {', '.join(hits)}")
                matches.append(b)
        if len(matches) != 1:
            sys.exit(
                f"{'No bucket' if not matches else 'Several buckets'} contain the date folders. "
                "Set BUCKET in .env to the right one."
            )
        bucket = matches[0]
    print(f"Bucket: {bucket}")

    if is_aws and not endpoint:
        # Re-create the client in the bucket's own region to avoid 301 redirects on downloads.
        loc = client.get_bucket_location(Bucket=bucket).get("LocationConstraint") or "us-east-1"
        client = boto3.client("s3", region_name=loc, config=CLIENT_CONFIG, **creds)
    return client, bucket


def list_objects(client, bucket, prefix):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        yield from page.get("Contents", [])


def download(client, bucket, obj, out_dir: Path):
    dest = out_dir / obj["Key"]
    if dest.exists() and dest.stat().st_size == obj["Size"]:
        return obj["Key"], "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    client.download_file(bucket, obj["Key"], str(tmp), Config=TRANSFER_CONFIG)
    tmp.rename(dest)
    return obj["Key"], "downloaded"


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def verify(out_dir: Path, sessions):
    """Check downloaded files against each session's SHA256SUMS.txt. Returns number of bad files."""
    bad = 0
    for sums in sorted(out_dir.glob("*/*/SHA256SUMS.txt")):
        session_dir = sums.parent
        if session_dir.name not in sessions:
            continue
        print(f"Verifying {session_dir.name} ({SESSIONS.get(session_dir.name, '?')})")
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            expected, rel = line.split(maxsplit=1)
            rel = rel.lstrip("*")  # binary-mode marker from sha256sum
            path = session_dir / rel
            if not path.exists():
                print(f"  missing  {rel}")  # e.g. skipped with --skip-mcap
            elif sha256(path) != expected.lower():
                print(f"  MISMATCH {rel}")
                bad += 1
            else:
                print(f"  ok       {rel}")
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="output directory (default: ./data)")
    parser.add_argument("--list", action="store_true", help="list matching objects without downloading")
    parser.add_argument("--all", action="store_true", help="take everything under the date folders, not just the listed sessions")
    parser.add_argument("--session", action="append", metavar="ID_OR_NAME",
                        help="only these sessions, by ID or name (e.g. --session bakery); repeatable")
    parser.add_argument("--skip-mcap", action="store_true", help="metadata/JSON only, skip the large .mcap recordings")
    parser.add_argument("--verify", action="store_true", help="check SHA256SUMS.txt after downloading (reads every file)")
    parser.add_argument("--workers", type=int, default=8, help="files downloaded in parallel (default 8)")
    args = parser.parse_args()

    sessions = dict(SESSIONS)
    if args.session:
        wanted = {s.lower() for s in args.session}
        sessions = {sid: name for sid, name in SESSIONS.items() if sid in wanted or name.lower() in wanted}
        if not sessions:
            sys.exit(f"No session matches {args.session}. Known: {', '.join(SESSIONS.values())}")

    client, bucket = make_client()

    objects = []
    for folder in DATE_FOLDERS:
        for obj in list_objects(client, bucket, f"{folder}/"):
            key = obj["Key"]
            if key.endswith("/") or (args.skip_mcap and key.endswith(".mcap")):
                continue
            if (args.all and not args.session) or any(sid in key for sid in sessions):
                objects.append(obj)

    found = {sid for sid in sessions if any(sid in o["Key"] for o in objects)}
    for sid, name in sessions.items():
        n = sum(1 for o in objects if sid in o["Key"])
        size = sum(o["Size"] for o in objects if sid in o["Key"])
        print(f"{'✓' if sid in found else '✗'} {sid} ({name}): {n} files, {human(size)}")
    total = sum(o["Size"] for o in objects)
    print(f"\n{len(objects)} objects, {human(total)} total")

    if missing := set(sessions) - found:
        print(f"WARNING: no objects found for {len(missing)} session(s): {', '.join(sorted(missing))}")
        if not objects:
            print("Top-level of bucket, to help locate them:")
            resp = client.list_objects_v2(Bucket=bucket, Delimiter="/")
            for p in resp.get("CommonPrefixes", []):
                print(f"  {p['Prefix']}")

    if args.list:
        for o in objects:
            print(f"  {human(o['Size']):>10}  {o['Key']}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = sum(
        o["Size"] for o in objects
        if not ((out_dir / o["Key"]).exists() and (out_dir / o["Key"]).stat().st_size == o["Size"])
    )
    free = shutil.disk_usage(out_dir).free
    print(f"Still to download: {human(remaining)}; free space at {out_dir.resolve()}: {human(free)}")
    if remaining > free:
        sys.exit("Not enough disk space. Use --out on a bigger drive, or --session to pull a subset.")

    done = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, client, bucket, o, out_dir): o for o in objects}
        for fut in as_completed(futures):
            done += 1
            try:
                key, status = fut.result()
                print(f"[{done}/{len(objects)}] {status}: {key}")
            except Exception as e:  # noqa: BLE001 - report and keep going
                key = futures[fut]["Key"]
                failures.append(key)
                print(f"[{done}/{len(objects)}] FAILED: {key}: {e}", file=sys.stderr)

    if failures:
        sys.exit(f"\n{len(failures)} download(s) failed; re-run to retry (completed files are skipped).")
    print(f"\nDone. Files in {out_dir.resolve()}")

    if args.verify and verify(out_dir, sessions):
        sys.exit("Checksum mismatches found; delete those files and re-run to re-download.")


if __name__ == "__main__":
    main()
