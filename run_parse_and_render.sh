#!/usr/bin/env bash
# Usage: ./run_parse_and_render.sh <image_path> <bpm>
set -euo pipefail

IMAGE_PATH=${1:-Peaceful-Easy-Feeling-Eagles-Drum-Sheet-Music.png}
BPM=${2:-120}

source .venv/bin/activate
IMAGE_PATH="$IMAGE_PATH" BPM="$BPM" python - <<'PY'
import json
import os
import tempfile
from music21 import instrument, note, stream, tempo
import main

image_path = os.environ["IMAGE_PATH"]
bpm = int(os.environ["BPM"])

if not os.path.exists(image_path):
    raise SystemExit(f"Image not found: {image_path}")
if not main.INFERENCE_RUNNER:
    raise SystemExit("Detector not initialized; set MODEL_WEIGHTS_PATH")

parsed = main._parse_image_with_model(image_path)
payload = {
    "filename": image_path,
    "bpm": bpm,
    "status": "parsing_successful",
    "parsed_notes": parsed,
    "source": "detector",
}
with open("parsed_notes.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)

score = stream.Stream()
score.append(tempo.MetronomeMark(number=bpm))
part = stream.Part()
part.insert(0, instrument.Percussion())
for note_data in payload["parsed_notes"]:
    midi = note_data.get("midi_pitch", 0)
    dur = note_data.get("duration", 1.0)
    offset = note_data.get("offset", 0.0)
    if midi == 0:
        n = note.Rest()
    else:
        n = note.Note()
        n.midi = midi
    n.duration.quarterLength = dur
    part.insert(offset, n)
score.append(part)

with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
    midi_path = tmp.name
score.write("midi", fp=midi_path)
main._render_with_fluidsynth(midi_path, "peaceful_take.wav")
os.remove(midi_path)
PY
