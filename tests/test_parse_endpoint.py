import io
import os

os.environ.setdefault("SKIP_MODEL_LOAD", "1")

from fastapi.testclient import TestClient
from PIL import Image

import main
from model_inference import Detection


def _fake_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_parse_endpoint_uses_detector(monkeypatch):
    client = TestClient(main.app)

    class DummyRunner:
        def predict_path(self, path):
            return [Detection(bbox=[0, 0, 10, 10], score=0.99, label=7)]

    def fake_detections_to_notes(detections, image_height, **_):
        assert len(detections) == 1
        assert image_height > 0
        return [
            {"midi_pitch": 51, "duration": 1.0, "offset": 0.0, "confidence": 0.99}
        ]

    monkeypatch.setattr(main, "INFERENCE_RUNNER", DummyRunner())
    monkeypatch.setattr(main, "detections_to_notes", fake_detections_to_notes)

    response = client.post(
        "/parse_drumsheet/?bpm=120",
        files={"file": ("test.png", _fake_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "detector"
    assert payload["parsed_notes"][0]["midi_pitch"] == 51
