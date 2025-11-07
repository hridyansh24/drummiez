# Drummiez AI

This project aims to use Artificial Intelligence to parse musical drum sheets (from PDF/image files) and then generate drum audio based on the extracted notes.

## Features

- **Drum Sheet Parsing:** AI-powered parsing of drum sheet images/PDFs to extract musical notes.
- **Drum Audio Generation:** Synthesis of drum sounds based on the parsed notes.
- **Configurable BPM:** Supports custom BPM settings, with a default of 100 BPM.
- **API Endpoints:** Provides a backend API for integration with a frontend application.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd drummiez
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install FluidSynth:**
    `midi2audio` relies on `FluidSynth` for audio generation. You need to install `FluidSynth` on your system.

    - **macOS:**
      ```bash
      brew install fluidsynth
      ```
    - **Debian/Ubuntu:**
      ```bash
      sudo apt-get update
      sudo apt-get install fluidsynth
      ```
    - **Windows:** Download from [FluidSynth's official website](https://www.fluidsynth.org/download/).

5.  **Obtain a Soundfont:**
    `FluidSynth` requires a soundfont (`.sf2` file) to produce sounds. You can download a General MIDI soundfont, for example, `FluidR3_GM.sf2`.
    Place the soundfont file in a known location, and set the `SOUNDFONT_PATH` environment variable to its absolute path. For example:
    ```bash
    export SOUNDFONT_PATH="/path/to/your/soundfont.sf2"
    ```
    A common path on Linux is `/usr/share/sounds/sf2/FluidR3_GM.sf2`.

6.  **Point the API at your trained detector weights (optional but recommended):**
    ```bash
    export MODEL_WEIGHTS_PATH="/absolute/path/to/drum_omr_model.pth"
    export MODEL_CONFIDENCE=0.5  # raise this if you want fewer detections
    ```
    If the environment variable is omitted, the app looks for `drum_omr_model.pth` in the repository root. When weights are available, `/parse_drumsheet/` can accept PNG/JPG/BMP/TIFF inputs and will convert the detections into a simple drum pattern.

7.  **(Optional) Provide a label-to-MIDI mapping:**
    ```bash
    export DRUM_LABEL_MAP_PATH="/absolute/path/to/label_map.json"
    ```
    The JSON should map detector label IDs (integers starting at 1) to MIDI pitches, for example:
    ```json
    {
      "1": 42,
      "2": 38,
      "3": 36
    }
    ```
    If omitted, the API falls back to a vertical-position heuristic (top = hi-hat, middle = snare, bottom = kick).

8.  **Run the application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    You can also run `python main.py`, which calls the same `uvicorn` command.

## End-to-end usage

1. **Parse a drum sheet image with the trained detector** (requires `MODEL_WEIGHTS_PATH`):
   ```bash
   curl -F "file=@/path/to/sheet.png" \
        "http://localhost:8000/parse_drumsheet/?bpm=110" \
        -o parsed_notes.json
   ```
   The response contains `parsed_notes`, each with a `midi_pitch`, `duration`, `offset`, and the model's confidence score. The current heuristic orders detections left-to-right and maps their vertical position to kick/snare/hi-hat sounds so you can quickly audition model outputs.

2. **Generate playable audio from the parsed notes:**
   ```bash
   curl -X POST "http://localhost:8000/generate_drum_audio/?bpm=110" \
        -H "Content-Type: application/json" \
        -d @parsed_notes.json \
        -o drummiez_take.wav
   open drummiez_take.wav  # or use any media player
   ```

3. **MusicXML fallback:** If you already have a MusicXML file (or the `oemer` runtime generates one from PDF), upload it to `/parse_drumsheet/` the same way; the endpoint will bypass the detector and directly parse the score via `music21`.

## Testing

Run the unit and API tests with:
```bash
pytest
```

## API Endpoints

- `GET /`: Welcome message.
- `POST /parse_drumsheet/`: Upload a drum sheet (image/PDF/MusicXML) for AI parsing. Accepts `bpm` as an optional query parameter.
- `POST /generate_drum_audio/`: Generate drum audio from provided notes. Accepts `bpm` as an optional query parameter.

## Technologies Used

- Python
- FastAPI (for API)
- Oemer (for Optical Music Recognition - currently a placeholder)
- music21 (for MusicXML parsing and MIDI generation)
- midi2audio (for MIDI to audio conversion)
- FluidSynth (external dependency for audio synthesis)

## Contributing

Contributions are welcome! Please refer to the contributing guidelines (TBD).

## License

This project is licensed under the MIT License (TBD).
