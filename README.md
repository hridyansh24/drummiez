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

6.  **Run the application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

## API Endpoints

- `GET /`: Welcome message.
- `POST /parse_drumsheet/`: Upload a drum sheet (image/PDF) for AI parsing. Accepts `bpm` as an optional query parameter.
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
