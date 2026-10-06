# MANO hand-mesh scene from Mudra EMG episodes — design

Date: 2026-10-06
Status: draft, awaiting review

## Goal

One script that takes an episode from `zander-eeg-data/09_21_26_POC/`, generates MANO hand meshes from its
egocentric video with WiLoR, and produces an Encord upload JSON for a single composite **scene** containing:

- the egocentric video (as a frame sequence)
- the left-hand mesh
- the right-hand mesh
- the EMG time series (left and right band)

This is a demo for Long Horizon. Presentation of the EMG only needs to be legible, not polished.

## Why a scene, not a data group

Data groups reject scene items (`DATA_GROUP_FORBIDDEN_TYPES` in
`backend/.../business_logic/storage/operations.py`, still true on master as of 2026-10-06), and meshes can
only be uploaded as scene `model` streams. Since August, scenes support `time_series` streams (#8926) and
tiled layouts (#9144), so one composite scene can hold every piece. Reference: Sebastian Harper's scene with
point clouds, a 3D model and time series in the Encord workspace on app.us.encord.com.

## Inputs

- Episode MCAP from Cloudflare R2, bucket `zander-eeg-data`, prefix `09_21_26_POC/`, listed in `manifest.json`
  (39 files: 21 task episodes, 18 EMG calibrations). Read with `wrangler r2 object get --remote` using the
  user's own Cloudflare login (`npx wrangler login`, Encord account); `.env` keys are no longer used.
- Topics used:
  - `/ego/zed_head/side_by_side/image_compressed` — `foxglove.CompressedVideo`, H.265, 3840x1200 (left|right), 60 fps
  - `/ego/zed_head/left/camera_info` — `sensor_msgs/CameraInfo`, 1920x1200 left-eye intrinsics + distortion
  - `/mudra/wristband_{left,right}/emg` — JSON `mudra.Sample`, 3 channels at 2133 Hz
- `log_time` is the recording box's UTC clock (ns) for every topic, so video and EMG share a clock.

## Outputs

Downloaded MCAP and its split, in `data/<uuid>/`:

```
<uuid>.mcap
extracted/summary.json             time range; per topic: schema, message count, output file
extracted/metadata.json            MCAP metadata records
extracted/<topic path>.<ext>       one file per topic, path mirrors the topic name:
                                   video -> .h265 + .frames.csv (log_time, byte offset, size) + .mp4 (remux),
                                   CameraInfo -> .json, Imu -> .csv, mudra.Sample -> .csv, other -> .jsonl
```

Scene working directory `out/<episode_id>/`, built from `extracted/`:

```
frames/<log_time_ns>.jpg          left eye, sampled at --fps
camera.json                       left-eye intrinsics + distortion
emg_left.csv, emg_right.csv       time column + v0..v2
mano.npz                          per frame, per hand: MANO pose, shape, cam translation, vertices, score
meshes/{left,right}/<log_time_ns>.obj
scene.json                        the scene definition
```

Uploaded to `gs://long-horizon-wristband-demo-data/<uuid>/` (the episode's R2 uuid, so every file a scene
references sits in one folder; same layout, minus `mano.npz`). After a
successful upload the scene is added (replaced by title) to `encord_upload.json` in the repo root, which
accumulates every uploaded episode; the user registers that one file in the Encord UI against the bucket's GCS
integration (Encord skips scenes it already has).

## Interface

```
uv run mano_scene.py                      # list episodes (uuid, episode_id, kind, duration, size)
uv run mano_scene.py <uuid-prefix | episode_id>
    [--fps 15]                            # frames sampled per second of video (default 15)
    [--start SECONDS] [--duration SECONDS]# optional clip window, relative to episode start
    [--device auto|cuda|mps|cpu]          # auto = cuda > mps > cpu
    [--no-upload]                         # stop after writing local files + JSON
```

Single self-contained `uv` script with inline dependencies. Previous scripts live in `legacy/` and are not
imported.

## Pipeline

Each stage writes its output into `out/<episode_id>/` and is skipped when that output already exists, so a
failed or interrupted run resumes without re-downloading or re-running WiLoR.

1. **Download + split.** Resolve the episode from `manifest.json`, download the MCAP from R2 to
   `data/<uuid>/`, verify `mcap_sha256`, split every topic into `data/<uuid>/extracted/` and remux the video to
   `.mp4`. Later stages read only `extracted/`.
2. **Extract.**
   - Decode the video topic with PyAV. Each MCAP message is one frame; keep every Nth frame
     (N = round(60 / fps)) within the clip window, crop the left half (1920x1200), save as JPEG named by the
     message `log_time`.
   - Write `camera.json` from the first left `camera_info` message.
   - Write `emg_left.csv` / `emg_right.csv` from the EMG topics, restricted to the clip window, with the time
     column in the unit the scene timeline uses (see Open questions 1).
3. **Hands.** Run WiLoR (`wilor-mini`) on each JPEG. For each frame keep at most one detection per side
   (WiLoR's `is_right` flag), choosing the highest-confidence one; this drops stray hands of other people in
   view. Save results to `mano.npz`.
4. **Meshes.** For each kept detection, write an OBJ (778 vertices, MANO's 1538 faces) in the left camera's
   coordinate frame (vertices + WiLoR camera translation). A frame where a hand isn't detected gets no OBJ
   for that hand, so the mesh disappears rather than freezing.
5. **Scene.** Build the scene JSON:
   - `camera` — camera-parameters stream from `camera.json`
   - `ego` — image stream, camera `camera`, one event per JPEG with its timestamp
   - `left_hand`, `right_hand` — model streams, one URI event per OBJ with its timestamp, in the camera's
     frame of reference
   - `emg_left`, `emg_right` — time-series streams pointing at the CSVs
   - layout: top row `ego` image tile | `3d` tile; below, `emg_left` and `emg_right` timeseries tiles
6. **Upload + JSON.** `gcloud storage rsync` the publishable files to
   `gs://long-horizon-wristband-demo-data/<uuid>/`; then add
   `{"title": <episode_id>, "scene": <scene>, "clientMetadata": <episode metadata>}` (with `gs://` URLs) to the
   repo-root `encord_upload.json` (`{"scenes": [...]}`, sorted by title, replacing any scene with the same title).
   Skipped with `--no-upload`, so the file only lists scenes whose files are in the bucket.

## Encord prerequisites (user)

- A GCS integration for `long-horizon-wristband-demo-data` in the target workspace.
- Org feature flag `TIME_SERIES` enabled (off by default; without it time-series uploads fail).
- The target environment must have scene layouts and scene time-series streams (on master since August;
  confirmed on app.us by the reference scene).
- MANO model file, if `wilor-mini` doesn't bundle it (free registration at mano.is.tue.mpg.de). Note that
  MANO's licence is non-commercial by default; check before using this beyond a demo.

## Compute

Primary target is the M4 Pro (48 GB) on the Apple GPU (MPS), with CPU fallback. Estimated 2–5 frames/s, so
a 60 s episode at 15 fps (~900 frames) takes a few minutes. For long episodes the same script runs on the ML
server's L4 GPUs (`--device cuda`).

## Testing

1. **Benchmark:** WiLoR on ~100 frames on this Mac; record frames/s and check MPS works. If it's far below
   2 frames/s, move WiLoR runs to the L4 server.
2. **Scene validation:** validate the generated scene against the backend's own input models
   (`InputScene` in `backend/.../modalities/scenes/api_models/api_input/`, from an up-to-date master
   checkout) before uploading, so format errors show up locally.
3. **End to end:** run on a short calibration episode (~12 s, e.g. `0aa8de95`), upload, register in Encord,
   and check: frames play, both hands appear in the 3D view and track the video, EMG tiles show both bands
   on the same timeline.
4. **Task episode:** repeat on a Glue episode (~70 s, e.g. `5c5be84a`) as the demo candidate.

## Open questions (resolve during build, by reading backend/frontend code or testing)

1. **Timestamp units.** What unit scene event timestamps use, and how a scene time-series CSV's time column
   maps onto the scene timeline. Use the same clock for frames, meshes and EMG; prefer nanoseconds or
   milliseconds relative to clip start if absolute UTC ns causes problems.
2. **Mesh alignment.** Whether model streams need an explicit frame of reference / pose to line up with the
   camera, and the camera axis convention (OpenCV vs OpenGL) the scene expects.
3. **EMG density.** Whether 2133 Hz x 3 channels is too dense for the timeseries tile. If so, add
   `--emg-hz` to decimate (e.g. RMS envelope at 100 Hz, which also reads better visually).
4. **MANO file.** Whether `wilor-mini` downloads everything it needs or needs `MANO_RIGHT.pkl` supplied.

### Resolutions (2026-10-06, from backend origin/master `3bf4d637e` and WiLoR-mini source)

1. Scene timestamps are strict integers and the viewer is frame-indexed: the first timestamp must be < 10, so
   absolute UTC ns won't load. All streams (and the CSV `time` column, which must be a non-negative integer)
   use **integer ms since the first saved frame**. Time-series samples past the last image timestamp fall
   outside the timeline, so EMG is cut to the span of the saved frames.
2. Model streams ignore frames of reference (meshes are drawn at the root), and a camera with no frame of
   reference sits at the root with identity pose. So: OBJ vertices in left-camera coordinates, and
   `worldConvention` = `cameraConvention` = `{x: right, y: down, z: forward}` (OpenCV). The viewer keeps a
   model until its next event, so a hand that disappears gets an event pointing at `meshes/empty.obj`
   (a degenerate triangle; an empty OBJ fails to parse).
3. ms timestamps make raw 2133 Hz samples share timestamps, so `--emg-hz` defaults to a 100 Hz RMS envelope
   (`--emg-hz 0` keeps raw samples).
4. WiLoR-mini downloads `MANO_RIGHT.pkl`, the detector and weights from Hugging Face (cached in
   `~/.cache/wilor-mini`). WiLoR's focal length is set so its full-image focal equals the real fx, and its
   translation is shifted from the image centre to the real principal point.

Still unverified (needs the UI): that the layout is applied, whether the `TIME_SERIES` flag gates the
time-series tiles in the frontend, and that the OBJ axes render as expected.

## Out of scope

- EMG-to-pose modelling.
- Triangulating hands from both eyes (single-eye WiLoR only).
- Using the wristband fiducial markers to fix wrist pose or filter hands.
- Temporal smoothing of MANO parameters (add later if the meshes jitter badly).
- Registering via the Encord SDK (the user uploads the JSON in the UI).
