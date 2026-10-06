"""Validate an encord_upload.json against the backend's scene upload models (same code path as the upload).

Needs a backend checkout that includes scene time series (#8926) and layouts (#9144). Run from ~/Localdev/backend:
    uv run --directory projects/api-server python ~/Localdev/emg-demo/validate_scene.py <encord_upload.json>

Backend-only: frontend rules (first timestamp < 10, integer CSV time column, one 3D tile) aren't checked here.
"""

import json
import sys

from cord.apiserver.business_logic.modalities.scenes.api_models.api_input.input_scene import DataUploadSceneDefinition
from cord.apiserver.business_logic.modalities.scenes.api_models.api_input.translate_input_to_internal import (
    translate_input_to_internal,
)

for item in json.load(open(sys.argv[1]))["scenes"]:
    scene = DataUploadSceneDefinition.model_validate(item)
    translate_input_to_internal(scene.scene)
    print("OK", scene.title)
