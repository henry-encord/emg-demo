"""Camera permission on macOS, asked directly through AVFoundation.

Qt's QCameraPermission needs NSCameraUsageDescription in the app's Info.plist, which a bare `python` doesn't have,
so Qt can never show the prompt. AVFoundation's own request works: macOS attributes it to the app that launched
Python (Terminal, iTerm, VS Code, ...) and shows the usual "<app> would like to access the camera" dialog. Once a
user has answered, macOS never asks again; a denial can only be undone in System Settings (and the launching app
usually has to be restarted to pick up the change).

Off macOS (or without PyObjC) everything reports "granted" so the camera source just tries to open the device.
"""

from __future__ import annotations

import sys
from typing import Callable, Literal

Status = Literal["granted", "denied", "restricted", "undetermined"]

SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"

try:
    if sys.platform != "darwin":
        raise ImportError
    import AVFoundation as _AV
except ImportError:
    _AV = None


def status() -> Status:
    if _AV is None:
        return "granted"
    s = _AV.AVCaptureDevice.authorizationStatusForMediaType_(_AV.AVMediaTypeVideo)
    return {
        _AV.AVAuthorizationStatusAuthorized: "granted",
        _AV.AVAuthorizationStatusDenied: "denied",
        _AV.AVAuthorizationStatusRestricted: "restricted",
    }.get(s, "undetermined")


def request(done: Callable[[bool], None]) -> None:
    """Show the macOS prompt if the user hasn't answered yet; `done(granted)` is called on an arbitrary thread
    (immediately if already answered)."""
    if _AV is None:
        done(True)
        return
    _AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(_AV.AVMediaTypeVideo, lambda ok: done(bool(ok)))
