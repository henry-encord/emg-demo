#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "python-dotenv", "mcap"]
# ///
"""Show what's in zander-eeg-data/09_21_26_POC/: folders, file types, and a peek inside one file of each type.

Usage:
    uv run scan_for_emg.py > emg_scan.txt
"""

import os
from collections import Counter, defaultdict

os.environ["BUCKET"] = "zander-eeg-data"
PREFIX = "09_21_26_POC/"

from inspect_bucket import inspect_mcap  # noqa: E402
from pull_sessions import human, list_objects, make_client  # noqa: E402

MAGIC = {
    b"0       ": "EDF/BDF",
    b"XDF:": "XDF (Lab Streaming Layer)",
    b"\x93NUMPY": "NumPy .npy",
    b"PK\x03\x04": "ZIP (or .npz)",
    b"\x89HDF": "HDF5",
    b"PAR1": "Parquet",
    b"\x89MCAP": "MCAP",
}


def ext_of(key):
    name = key.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else "(none)"


def peek(client, bucket, obj):
    print(f"\n--- {obj['Key']} ({human(obj['Size'])}) ---")
    if obj["Key"].endswith(".mcap"):
        inspect_mcap(client, bucket, obj)
        return
    head = client.get_object(Bucket=bucket, Key=obj["Key"], Range="bytes=0-2047")["Body"].read()
    kind = next((v for k, v in MAGIC.items() if head.startswith(k)), None)
    if kind == "EDF/BDF":
        n = int(head[252:256])
        labels = [head[256 + i * 16: 272 + i * 16].decode(errors="replace").strip() for i in range(min(n, 110))]
        print(f"  EDF/BDF, {n} signals: {labels}")
    elif kind:
        print(f"  {kind}")
        print("  " + head[:300].decode(errors="replace").replace("\n", "\n  "))
    else:
        try:
            print("  " + "\n  ".join(head.decode().splitlines()[:15]))
        except UnicodeDecodeError:
            print(f"  binary, first bytes: {head[:64].hex(' ')}")


def main():
    client, bucket = make_client()
    objs = [o for o in list_objects(client, bucket, PREFIX) if not o["Key"].endswith("/")]
    print(f"\n{len(objs)} files, {human(sum(o['Size'] for o in objs))} under {bucket}/{PREFIX}")

    print("\nFolders:")
    folders = defaultdict(list)
    for o in objs:
        folders[o["Key"].rsplit("/", 1)[0] + "/"].append(o)
    for folder, files in sorted(folders.items()):
        exts = Counter(ext_of(f["Key"]) for f in files)
        print(f"  {folder:<70} {len(files):>5} files {human(sum(f['Size'] for f in files)):>10}  {dict(exts)}")

    print("\nFile types:", dict(Counter(ext_of(o["Key"]) for o in objs).most_common()))

    print("\nOne sample of each file type:")
    seen = set()
    for o in objs:
        if ext_of(o["Key"]) not in seen:
            seen.add(ext_of(o["Key"]))
            peek(client, bucket, o)


if __name__ == "__main__":
    main()
