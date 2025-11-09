"""
Utilities for running the trained Faster R-CNN drum detector and converting
its detections into simplified drum note events that the API can play back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from importlib import import_module

from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: List[float]
    score: float
    label: int


class DrumOMRInference:
    """
    Loads the Faster R-CNN detector trained via train_model.py and exposes
    helper methods to run predictions on sheet images.
    """

    def __init__(
        self,
        weights_path: str,
        detection_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        if not weights_path:
            raise ValueError("weights_path must be provided")

        torch_module = _load_torch()
        self._torch = torch_module
        self.device = torch_module.device(
            device if device else ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        self.detection_threshold = detection_threshold
        self.model = self._build_model()
        LOGGER.info("Loading detector weights from %s", weights_path)
        state_dict = torch_module.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _build_model(num_classes: int = 2):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def predict_image(self, image: Image.Image) -> List[Detection]:
        from torchvision.transforms.functional import to_tensor

        tensor = to_tensor(image).to(self.device)
        torch_module = self._torch
        with torch_module.no_grad():
            output = self.model([tensor])[0]

        boxes = output.get("boxes")
        scores = output.get("scores")
        labels = output.get("labels")

        if boxes is None or scores is None:
            return []

        if labels is None:
            labels = torch_module.ones(len(boxes), dtype=torch_module.int64, device=self.device)

        detections: List[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            score_value = float(score.item())
            if score_value < self.detection_threshold:
                continue
            detections.append(
                Detection(
                    bbox=box.tolist(),
                    score=score_value,
                    label=int(label.item()),
                )
            )

        return detections

    def predict_path(self, path: str) -> List[Detection]:
        with Image.open(path) as img:
            rgb_image = img.convert("RGB")
            return self.predict_image(rgb_image)


def detections_to_notes(
    detections: Iterable[Detection],
    image_height: float,
    duration: float = 1.0,
    label_to_midi: Optional[Dict[int, int]] = None,
) -> List[dict]:
    """
    Simplified heuristic that turns detections into sequential drum notes.
    - Notes are ordered from left to right.
    - label_to_midi can override the default vertical-position mapping.
    """

    sorted_dets = sorted(detections, key=lambda det: det.bbox[0])
    parsed_notes: List[dict] = []
    label_to_midi = label_to_midi or {}

    if image_height <= 0:
        LOGGER.warning("Invalid image height %s; defaulting to 1.0 for normalization", image_height)
        image_height = 1.0

    for idx, det in enumerate(sorted_dets):
        midi_pitch = label_to_midi.get(det.label)
        if midi_pitch is None:
            y_min, y_max = det.bbox[1], det.bbox[3]
            y_center_norm = ((y_min + y_max) / 2.0) / image_height
            midi_pitch = _midi_from_vertical_position(y_center_norm)

        parsed_notes.append(
            {
                "midi_pitch": midi_pitch,
                "duration": duration,
                "offset": idx * duration,
                "confidence": det.score,
                "label": det.label,
            }
        )

    return parsed_notes


def load_label_mapping(json_path: str) -> Dict[int, int]:
    """
    Reads a JSON file mapping detector label IDs to MIDI pitches.
    Expected format: { "1": 42, "2": 38 }
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Label mapping JSON must be a dictionary")

    mapping: Dict[int, int] = {}
    for key, value in data.items():
        try:
            label_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label id '{key}' in mapping") from exc

        if isinstance(value, dict):
            midi_value = value.get("midi")
        else:
            midi_value = value

        if midi_value is None:
            raise ValueError(f"Missing MIDI value for label '{key}'")

        mapping[label_id] = int(midi_value)

    return mapping


def _midi_from_vertical_position(y_center_norm: float) -> int:
    """
    Maps a normalized vertical staff position to an approximate drum sound.
    """
    if y_center_norm < 0.33:
        return 42  # Closed hi-hat
    if y_center_norm < 0.66:
        return 38  # Snare
    return 36  # Kick


def _load_torch():
    try:
        return import_module("torch")
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("PyTorch is required for DrumOMRInference") from exc
