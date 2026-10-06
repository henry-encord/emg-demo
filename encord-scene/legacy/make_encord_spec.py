#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "python-dotenv"]
# ///
"""Build an Encord JSON upload spec that registers the session MCAPs in place (no download).

Each *_merged.mcap becomes one Scene. Its sibling <hash>_<time>.json is attached as clientMetadata,
together with the session ID, session name and date.

Usage:
    uv run make_encord_spec.py                      # -> encord_upload.json
    uv run make_encord_spec.py --session bakery     # subset
    uv run make_encord_spec.py --no-metadata        # skip fetching the per-recording JSON
"""

import argparse
import json
import sys
from pathlib import PurePosixPath

from pull_sessions import DATE_FOLDERS, SESSIONS, list_objects, make_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="encord_upload.json")
    parser.add_argument("--session", action="append", metavar="ID_OR_NAME", help="only these sessions; repeatable")
    parser.add_argument("--no-metadata", action="store_true", help="don't attach the per-recording JSON as clientMetadata")
    args = parser.parse_args()

    sessions = dict(SESSIONS)
    if args.session:
        wanted = {s.lower() for s in args.session}
        sessions = {sid: name for sid, name in SESSIONS.items() if sid in wanted or name.lower() in wanted}
        if not sessions:
            sys.exit(f"No session matches {args.session}. Known: {', '.join(SESSIONS.values())}")

    client, bucket = make_client()
    endpoint = client.meta.endpoint_url.rstrip("/")  # must match the endpoint of the Encord integration

    keys = {o["Key"] for folder in DATE_FOLDERS for o in list_objects(client, bucket, f"{folder}/")}

    scenes = []
    for key in sorted(keys):
        if not key.endswith("_merged.mcap"):
            continue
        sid = next((s for s in sessions if f"/{s}/" in key), None)
        if not sid:
            continue
        path = PurePosixPath(key)
        date, time = path.parts[0], path.parent.parent.name  # <date>/<sid>/raw/<date>/<sid>/<HH_MM_SS>/<hash>/<file>
        metadata = {"session_id": sid, "session_name": sessions[sid], "date": date, "segment_start": time}

        sidecar = key.removesuffix("_merged.mcap") + ".json"
        if not args.no_metadata and sidecar in keys:
            body = client.get_object(Bucket=bucket, Key=sidecar)["Body"].read()
            try:
                metadata["recording"] = json.loads(body)
            except json.JSONDecodeError:
                print(f"WARNING: {sidecar} isn't valid JSON, skipping it", file=sys.stderr)

        scenes.append({
            "title": f"{sessions[sid]} {date} {time.replace('_', ':')} ({path.parent.name[:8]})",
            "scene": {"url": f"{endpoint}/{bucket}/{key}", "format": "mcap"},
            "clientMetadata": metadata,
        })
        print(f"+ {scenes[-1]['title']}")

    with open(args.out, "w") as f:
        json.dump({"scenes": scenes}, f, indent=2)
    print(f"\nWrote {len(scenes)} scenes to {args.out}")


if __name__ == "__main__":
    main()
