# emg-demo

Two projects, each with its own environment (managed with [uv](https://docs.astral.sh/uv/)):

- `hand-viewer/` is a desktop app. It shows the video on the left and a 3D hand on the right.
- `encord-scene/` turns a recorded EMG episode into an Encord scene. It also writes the episodes that the viewer
  replays.

Requirements: an Apple-silicon Mac (or a Linux machine with CUDA), and Python 3.11, which uv installs for you.

## Hand viewer

```sh
cd hand-viewer
uv sync
uv run python -m hand_viewer
```

That starts the live camera. Pick another source with `--source`:

| Source | What it shows |
| --- | --- |
| `camera` | Live webcam, with hands from WiLoR. This is the default. macOS asks for camera access the first time. |
| `replay` | A recorded episode from `encord-scene/out/` (the newest one, or `--episode DIR`). |
| `replay-wilor` | A recorded episode's video, run through the live WiLoR path. |
| `synthetic` | Animated test hands. Needs no camera, episode or WiLoR models. |

Other options: `--camera N` picks a camera, and `--device mps|cuda|cpu` sets where WiLoR runs. The cog in the
top-right of the window shows or hides the controls.

**First run.** The first time WiLoR runs (`camera` or `replay-wilor`), it downloads its models (~2.4 GB) to
`~/.cache/wilor-mini`. The SOMA hand model (~20 MB) also downloads from Hugging Face on first use. So the first
start takes a while, and later starts take a few seconds. Plain `replay` uses the MANO model that WiLoR downloads,
so run `camera` (or `encord-scene`) once first. The first replay of an episode spends a few seconds converting
before the hands appear; after that, the result is cached.

Tests: `uv run pytest`.

## Encord scene

```sh
cd encord-scene
uv sync
uv run mano_scene.py                                # list episodes
uv run mano_scene.py sub-P001Fer_task-Glue_ep-005   # build one (add --no-upload to keep it local)
```

This downloads episodes from Cloudflare R2, so run `npx wrangler login` once first. It uploads with `gcloud`.
Output goes to `encord-scene/out/<episode>/`. See `encord-scene/2026-10-06-mano-scene-design.md` for details.
