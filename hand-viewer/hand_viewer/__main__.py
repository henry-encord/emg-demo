"""Live hand viewer: video on the left, the MANO hand render on the right.

Usage:
    uv run python -m hand_viewer [--source camera|replay|replay-wilor|synthetic] [--episode DIR]
                                 [--device auto|mps|cuda|cpu] [--camera INDEX] [--smoke-seconds N]

replay-wilor plays the recorded video through the live WiLoR path (camera mode without a camera).

Replay and synthetic start fast; torch-heavy WiLoR is only imported when the camera source is picked.
`--smoke-seconds N` quits after N seconds and prints frame/pose/render counts (for scripted checks).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m hand_viewer", description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=("camera", "replay", "replay-wilor", "synthetic"), default="replay")
    ap.add_argument("--episode", type=Path, help="encord-scene/out/<episode> dir (default: newest with mano.npz)")
    ap.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto", help="for WiLoR")
    ap.add_argument("--camera", type=int, help="camera index in the toolbar's list (0 = system default)")
    ap.add_argument("--smoke-seconds", type=float, help="quit after N seconds and print counts")
    args = ap.parse_args()

    # matplotlib/ultralytics (pulled in by WiLoR) write config dirs on import; keep them out of $HOME if unset.
    cache = Path(tempfile.gettempdir()) / "hand-viewer"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "mpl"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "yolo"))

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from hand_viewer.ui.main_window import MainWindow, warn_no_mano

    from hand_viewer.ui import theme

    app = QApplication(sys.argv)
    app.setApplicationName("Encord hand viewer")
    theme.apply(app)
    win = MainWindow(source=args.source, episode=args.episode, device=args.device, camera=args.camera)
    win.show()
    if args.smoke_seconds is not None:
        def finish():
            print("smoke:", " ".join(f"{k}={v}" for k, v in win.counts.items()), flush=True)
            win.close()
            app.quit()
        QTimer.singleShot(int(args.smoke_seconds * 1000), finish)
    else:
        warn_no_mano(win, win)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
