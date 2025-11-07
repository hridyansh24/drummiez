import math

from model_inference import Detection, detections_to_notes


def test_label_mapping_overrides_vertical_mapping():
    detections = [
        Detection(bbox=[0.0, 900.0, 50.0, 950.0], score=0.95, label=5),
    ]

    notes = detections_to_notes(
        detections,
        image_height=1000.0,
        duration=0.5,
        label_to_midi={5: 49},
    )

    assert len(notes) == 1
    assert math.isclose(notes[0]["duration"], 0.5)
    assert notes[0]["midi_pitch"] == 49
    assert notes[0]["label"] == 5


def test_vertical_mapping_used_when_label_missing():
    detections = [
        Detection(bbox=[0.0, 50.0, 40.0, 100.0], score=0.9, label=1),
        Detection(bbox=[60.0, 400.0, 120.0, 460.0], score=0.9, label=2),
        Detection(bbox=[140.0, 800.0, 200.0, 880.0], score=0.9, label=3),
    ]

    notes = detections_to_notes(detections, image_height=900.0)

    assert [note["midi_pitch"] for note in notes] == [42, 38, 36]
