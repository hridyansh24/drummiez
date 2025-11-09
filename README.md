# Drummiez AI 🥁

> *Turning messy drum sheets into playable WAVs so you don’t have to squint at notation at 2 a.m.*

Think of Drummiez as my “I actually shipped this” college capstone: a FastAPI service that slurps PDFs or images, runs Optical Music Recognition, and spits out structured drum notes plus audio previews. I built it so I could nerd out on CV + audio, and now it doubles as a portfolio piece when hiring managers peek at my GitHub.

Screens, demos, and memes will live here soon—right now you’re getting the engine room.

---

## Why it’s cool

- **Detector ➜ MIDI ➜ Audio pipeline**: Faster R-CNN (PyTorch) for staff detection, optional `oemer` → MusicXML fallback, then `music21` + `midi2audio` with FluidSynth to render WAVs.
- **API-first build**: Everything is exposed through FastAPI endpoints, so it plugs straight into a web UI, CLI, or notebook.
- **Configurable musician brain**: Bring your own weights, SoundFont, BPM, and label maps via env vars—no code edits required.
- **College-budget friendly**: Pure Python (3.10+), no GPUs required for inference, works on my M2 laptop without meltdown.
- **Tested (when I let them track)**: Pytest suite for the parser + inference helpers; I keep them locally to avoid shipping sample data.

---

## Architecture tour

```
Upload (PDF/PNG/JPG)
     │
     ├── Detector engine (Faster R-CNN) ──► Label map ──► Parsed notes JSON
     │
     └── OEMER pipeline ──► MusicXML ──► music21 ──► MIDI notes
                                     │
                                     └── midi2audio + FluidSynth ──► WAV preview
```

- `main.py`: FastAPI endpoints, engine selection logic, MIDI mapping, audio rendering.
- `model_inference.py`: wraps the trained detector, converts detections to note events, loads label mappings.
- `prepare_dataset.py` / `train_model.py`: scripts for dataset wrangling + model training if you want to level up the detector.
- `data/` + `tests/`: ignored by GitHub; I keep local samples, CSVs, and pytest cases without leaking them.

---

## Quick hit endpoints

| Method | Path | What it does | Notes |
| ------ | ---- | ------------ | ----- |
| GET | `/` | sanity ping | returns a friendly JSON banner |
| POST | `/parse_drumsheet/` | upload a PDF/image/MusicXML, get structured drum notes | query params: `bpm`, `engine=auto|detector|oemer` |
| POST | `/generate_drum_audio/` | send parsed notes JSON, get a streaming WAV response | uses `SOUNDFONT_PATH` for tone flavor |

### Example flow

```bash
curl -F "file=@samples/snare_groove.png" \
     "http://localhost:8000/parse_drumsheet/?bpm=110" \
     -o parsed_notes.json

curl -X POST "http://localhost:8000/generate_drum_audio/?bpm=110" \
     -H "Content-Type: application/json" \
     -d @parsed_notes.json \
     -o groove_take.wav
open groove_take.wav
```

---

## Local setup

```bash
git clone <your-fork-url>
cd drummiez
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # includes FastAPI, Uvicorn, music21, midi2audio, PyTorch, etc.
brew install fluidsynth          # or apt-get / choco install, depending on OS
```

Environment knobs (drop these in `.env` or export them):

```bash
export SOUNDFONT_PATH="/absolute/path/to/FluidR3_GM.sf2"
export MODEL_WEIGHTS_PATH="/absolute/path/to/drum_omr_model.pth"
export MODEL_CONFIDENCE=0.6              # tweak detection strictness
export DRUM_LABEL_MAP_PATH="/path/to/label_map.json"
export SKIP_MODEL_LOAD=0                 # set to 1 to skip detector
```

Run it:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# or
python main.py
```

---

## Development notes

- **Testing**: `pytest` (kept local because fixtures include paid datasets). Feel free to un-ignore `tests/` if you add your own.
- **Formatting/Linting**: vanilla Python for now—Black/ruff configs are on the wishlist.
- **Data**: `prepare_dataset.py` expects DeepScoresV2 or your own drum sheets; saved CSVs live under `data/` and are ignored by Git.
- **SoundFonts**: I default to `FluidR3_GM.sf2`; drop any GM-compatible `.sf2` and point `SOUNDFONT_PATH` at it.

---

## Roadmap (a.k.a. what’s brewing)

1. Dial in percussion-specific parsing inside `music21` (notehead heuristics are good but not perfect).
2. Expand the MIDI map for ghost notes, rimshots, and all the spicy percussion toys.
3. Ship a lightweight React UI with drag-and-drop uploads + instant audio preview.
4. Containerize the whole thing (multi-stage Dockerfile + GPU optional build).
5. CI/CD via GitHub Actions once I stop hoarding private test fixtures.

If you’re a hiring manager skimming this: yes, I own the full stack here (modeling ➜ backend ➜ tooling). Happy to walk through the code or nerd out about why FluidSynth still slaps in 2025.

---

## License / Contact

MIT-ish (formal license incoming). Ping me on GitHub issues or LinkedIn if you want to collab, jam, or send me better drum grooves.
