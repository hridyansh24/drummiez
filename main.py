# Dataset for Optical Music Recognition (OMR)
# DeepScoresV2: https://zenodo.org/record/4782213
# This dataset can be used to train a model for drum sheet music recognition.

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional
import importlib
import os
import tempfile
import shutil
from uuid import uuid4
from music21 import converter, instrument, note, stream, tempo, chord
from midi2audio import FluidSynth
from PIL import Image

from model_inference import DrumOMRInference, detections_to_notes, load_label_mapping

try:
    _oemer_module = importlib.import_module("oemer")
    OEMER_RUNNER = getattr(_oemer_module, "run", None)
except ImportError:
    OEMER_RUNNER = None

app = FastAPI()
LOGGER = logging.getLogger("drummiez")

# General MIDI Drum Map
DRUM_MIDI_MAP = {
    # Bass Drums
    "acoustic bass drum": 35, "bass drum 1": 36, "kick": 36,
    # Snare Drums
    "acoustic snare": 38, "electric snare": 40, "snare": 38,
    # Toms
    "low floor tom": 41, "high floor tom": 43, "low tom": 45,
    "low-mid tom": 47, "hi-mid tom": 48, "high tom": 50,
    # Hi-Hats
    "closed hi-hat": 42, "closed hi hat": 42,
    "pedal hi-hat": 44, "pedal hi hat": 44,
    "open hi-hat": 46, "open hi hat": 46,
    # Cymbals
    "crash cymbal 1": 49, "crash 1": 49,
    "ride cymbal 1": 51, "ride 1": 51,
    "chinese cymbal": 52,
    "ride bell": 53,
    "tambourine": 54,
    "splash cymbal": 55,
    "cowbell": 56,
    "crash cymbal 2": 57, "crash 2": 57,
    "ride cymbal 2": 59, "ride 2": 59,
    # Other
    "side stick": 37,
    "hand clap": 39,
}

def get_midi_pitch(n: note.NotRest) -> int:
    """
    Determines the MIDI pitch for a music21 note based on its instrument,
    display step, and notehead.
    """
    instrument_candidate = None
    if hasattr(n, "getInstrument"):
        inst = n.getInstrument(returnDefault=False)
        if inst and getattr(inst, "instrumentName", None):
            instrument_candidate = inst.instrumentName.lower()
    if not instrument_candidate and getattr(n, "instrument", None):
        name = getattr(n.instrument, "instrumentName", None)
        if name:
            instrument_candidate = name.lower()
    if instrument_candidate and instrument_candidate in DRUM_MIDI_MAP:
        return DRUM_MIDI_MAP[instrument_candidate]

    # Try to get the instrument from the unpitched display step
    display_step = None
    if isinstance(n, note.Unpitched):
        display_step = (n.displayStep or "").lower()
    elif getattr(n, "unpitched", None) and getattr(n.unpitched, "displayStep", None):
        display_step = n.unpitched.displayStep.lower()
    if display_step and display_step in DRUM_MIDI_MAP:
        return DRUM_MIDI_MAP[display_step]

    # Try to guess based on notehead
    notehead_value = getattr(n, "notehead", None)
    if notehead_value:
        notehead_type = str(notehead_value).lower()
        if "x" in notehead_type:
            return DRUM_MIDI_MAP.get("closed hi-hat")  # A common default for 'x'
        if "circle-x" in notehead_type:
            return DRUM_MIDI_MAP.get("open hi-hat")

    # Default to acoustic bass drum if no mapping is found
    return 35

# Placeholder for soundfont path - USER NEEDS TO PROVIDE A VALID PATH TO A .sf2 FILE
SOUNDFONT_PATH = os.getenv("SOUNDFONT_PATH", "/usr/share/sounds/sf2/FluidR3_GM.sf2") # Common path on Linux, might differ on macOS/Windows
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "drum_omr_model.pth")
MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", "0.5"))
SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
DRUM_LABEL_MAP_PATH = os.getenv("DRUM_LABEL_MAP_PATH")
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD") == "1"

LABEL_TO_MIDI = {}
if DRUM_LABEL_MAP_PATH and os.path.exists(DRUM_LABEL_MAP_PATH):
    try:
        LABEL_TO_MIDI = load_label_mapping(DRUM_LABEL_MAP_PATH)
        LOGGER.info("Loaded label-to-MIDI mapping from %s", DRUM_LABEL_MAP_PATH)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Failed to load label mapping: %s", exc)

INFERENCE_RUNNER = None
if SKIP_MODEL_LOAD:
    LOGGER.info("SKIP_MODEL_LOAD is enabled; detector will not be initialized.")
elif os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        INFERENCE_RUNNER = DrumOMRInference(
            MODEL_WEIGHTS_PATH, detection_threshold=MODEL_CONFIDENCE
        )
        LOGGER.info("Loaded detector weights from %s", MODEL_WEIGHTS_PATH)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Failed to load detector weights: %s", exc)
else:
    LOGGER.info("Model weights not found at %s; image parsing disabled", MODEL_WEIGHTS_PATH)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Drummiez AI API!"}

@app.post("/parse_drumsheet/")
async def parse_drumsheet(file: UploadFile = File(...), bpm: Optional[int] = 100):
    # Save the uploaded file to a temporary location
    musicxml_content = None
    try:
        _, original_ext = os.path.splitext(file.filename or "")
        original_ext = original_ext or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        ext = original_ext.lower()

        if (
            OEMER_RUNNER
            and ext not in (".xml", ".musicxml")
            and not _is_supported_image(ext)
        ):
            output_dir = tempfile.mkdtemp()
            try:
                OEMER_RUNNER(tmp_file_path, output_path=output_dir)
                musicxml_files = [f for f in os.listdir(output_dir) if f.endswith('.musicxml')]
                if not musicxml_files:
                    raise HTTPException(status_code=500, detail="Oemer did not generate a MusicXML file.")
                with open(os.path.join(output_dir, musicxml_files[0]), 'r', encoding="utf-8") as f:
                    musicxml_content = f.read()
            finally:
                shutil.rmtree(output_dir)
                os.remove(tmp_file_path)
        else:
            if ext in (".xml", ".musicxml"):
                try:
                    musicxml_content = contents.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise HTTPException(status_code=400, detail=f"Unable to decode MusicXML file: {exc}") from exc
            elif _is_supported_image(ext):
                parsed_notes = _parse_image_with_model(tmp_file_path)
                return {
                    "filename": file.filename,
                    "bpm": bpm,
                    "status": "parsing_successful",
                    "parsed_notes": parsed_notes,
                    "source": "detector",
                }
            else:
                raise HTTPException(
                    status_code=501,
                    detail="Upload a MusicXML file or a supported image type (png, jpg, jpeg, bmp, tiff)."
                )

        # Parse MusicXML with music21
        score = converter.parse(musicxml_content, format='musicxml')

        # Extract drum notes with improved logic
        parsed_notes = []
        for part in score.parts:
            if isinstance(part.getInstrument(), instrument.Percussion):
                for item in part.flat.notesAndRests:
                    if isinstance(item, note.Rest):
                        parsed_notes.append({
                            "midi_pitch": 0,
                            "duration": float(item.duration.quarterLength),
                            "offset": float(item.offset)
                        })
                        continue

                    if isinstance(item, chord.Chord):
                        note_iterable = item.notes
                    else:
                        note_iterable = [item]

                    for playable in note_iterable:
                        if isinstance(playable, (note.Note, note.Unpitched)):
                            midi_pitch = get_midi_pitch(playable)
                            parsed_notes.append({
                                "midi_pitch": midi_pitch,
                                "duration": float(playable.duration.quarterLength),
                                "offset": float(playable.offset)
                            })

        return {"filename": file.filename, "bpm": bpm, "status": "parsing_successful", "parsed_notes": parsed_notes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing drum sheet: {e}")
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.post("/generate_drum_audio/")
async def generate_drum_audio(background_tasks: BackgroundTasks, parsed_notes: dict, bpm: Optional[int] = 100):
    if not os.path.exists(SOUNDFONT_PATH):
        raise HTTPException(status_code=500, detail=f"Soundfont not found at {SOUNDFONT_PATH}. Please set SOUNDFONT_PATH environment variable or install a soundfont.")

    try:
        # Create a music21 stream from the parsed notes
        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=bpm))

        drum_part = stream.Part()
        drum_part.insert(0, instrument.Percussion())

        for n_data in parsed_notes.get("parsed_notes", []):
            midi_pitch = n_data.get("midi_pitch")
            duration = n_data.get("duration")
            offset = n_data.get("offset")

            if midi_pitch == 0:
                n = note.Rest()
            else:
                n = note.Note()
                n.midi = midi_pitch

            n.duration.quarterLength = duration
            drum_part.insert(offset, n)

        s.append(drum_part)

        # Create temporary MIDI and WAV files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as midi_tmp:
            midi_file_path = midi_tmp.name
            s.write('midi', fp=midi_file_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
            wav_file_path = wav_tmp.name
            fs = FluidSynth(SOUNDFONT_PATH)
            fs.midi_to_audio(midi_file_path, wav_file_path)

        def audio_stream(path: str):
            with open(path, "rb") as audio_file:
                while True:
                    chunk = audio_file.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        filename = f"drummiez_{uuid4().hex}.wav"
        response = StreamingResponse(audio_stream(wav_file_path), media_type="audio/wav", headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        })

        def cleanup_file(path: str):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        if background_tasks is not None:
            background_tasks.add_task(cleanup_file, wav_file_path)
        else:
            cleanup_file(wav_file_path)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating drum audio: {e}")
    finally:
        if 'midi_file_path' in locals() and os.path.exists(midi_file_path):
            os.remove(midi_file_path)
        # The WAV file is returned, so we don't delete it immediately.
        # A more robust solution would involve serving it or cleaning it up after a download.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def _is_supported_image(extension: str) -> bool:
    return extension.lower() in SUPPORTED_IMAGE_EXT


def _parse_image_with_model(image_path: str):
    if not INFERENCE_RUNNER:
        raise HTTPException(
            status_code=503,
            detail="Detector is not available. Ensure MODEL_WEIGHTS_PATH points to drum_omr_model.pth",
        )

    detections = INFERENCE_RUNNER.predict_path(image_path)
    if not detections:
        raise HTTPException(status_code=422, detail="No drum glyphs detected in the provided image.")

    with Image.open(image_path) as img:
        parsed_notes = detections_to_notes(
            detections,
            img.height,
            label_to_midi=LABEL_TO_MIDI if LABEL_TO_MIDI else None,
        )

    return parsed_notes
