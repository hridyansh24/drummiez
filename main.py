# Dataset for Optical Music Recognition (OMR)
# DeepScoresV2: https://zenodo.org/record/4782213
# This dataset can be used to train a model for drum sheet music recognition.

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
import os
import tempfile
import shutil
from music21 import converter, instrument, note, stream, tempo
from midi2audio import FluidSynth
from oemer import oemer

app = FastAPI()

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

def get_midi_pitch(n: note.Note) -> int:
    """
    Determines the MIDI pitch for a music21 note based on its instrument,
    display step, and notehead.
    """
    # Try to get the instrument name from the note itself
    if n.instrument:
        instrument_name = n.instrument.instrumentName.lower()
        if instrument_name in DRUM_MIDI_MAP:
            return DRUM_MIDI_MAP[instrument_name]

    # Try to get the instrument from the unpitched display step
    if n.unpitched and n.unpitched.displayStep:
        display_step = n.unpitched.displayStep.lower()
        if display_step in DRUM_MIDI_MAP:
            return DRUM_MIDI_MAP[display_step]

    # Try to guess based on notehead
    if n.notehead:
        notehead_type = n.notehead.text.lower()
        if "x" in notehead_type:
            return DRUM_MIDI_MAP.get("closed hi-hat") # A common default for 'x'
        if "circle-x" in notehead_type:
            return DRUM_MIDI_MAP.get("open hi-hat")

    # Default to acoustic bass drum if no mapping is found
    return 35

# Placeholder for soundfont path - USER NEEDS TO PROVIDE A VALID PATH TO A .sf2 FILE
SOUNDFONT_PATH = os.getenv("SOUNDFONT_PATH", "/usr/share/sounds/sf2/FluidR3_GM.sf2") # Common path on Linux, might differ on macOS/Windows

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Drummiez AI API!"}

@app.post("/parse_drumsheet/")
async def parse_drumsheet(file: UploadFile = File(...), bpm: Optional[int] = 100):
    # Save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        # Create a temporary directory for Oemer's output
        output_dir = tempfile.mkdtemp()

        try:
            # Run Oemer to process the drum sheet
            # This assumes oemer.run() takes the input file path and an output directory
            oemer.run(tmp_file_path, output_path=output_dir)

            # Find the generated MusicXML file in the output directory
            musicxml_files = [f for f in os.listdir(output_dir) if f.endswith('.musicxml')]
            if not musicxml_files:
                raise HTTPException(status_code=500, detail="Oemer did not generate a MusicXML file.")

            # Read the MusicXML content
            with open(os.path.join(output_dir, musicxml_files[0]), 'r') as f:
                musicxml_content = f.read()
        finally:
            # Clean up the temporary directories
            shutil.rmtree(output_dir)
            os.remove(tmp_file_path)

        # Parse MusicXML with music21
        score = converter.parse(musicxml_content, format='musicxml')

        # Extract drum notes with improved logic
        parsed_notes = []
        for part in score.parts:
            # Check if the part is a percussion part
            if isinstance(part.getInstrument(), instrument.Percussion):
                for item in part.flat.notesAndRests:
                    if isinstance(item, note.Note):
                        midi_pitch = get_midi_pitch(item)
                        parsed_notes.append({
                            "midi_pitch": midi_pitch,
                            "duration": float(item.duration.quarterLength),
                            "offset": float(item.offset)
                        })
                    elif isinstance(item, note.Rest):
                        parsed_notes.append({
                            "midi_pitch": 0,  # 0 for rests
                            "duration": float(item.duration.quarterLength),
                            "offset": float(item.offset)
                        })

        return {"filename": file.filename, "bpm": bpm, "status": "parsing_successful", "parsed_notes": parsed_notes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing drum sheet: {e}")
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.post("/generate_drum_audio/")
async def generate_drum_audio(parsed_notes: dict, bpm: Optional[int] = 100):
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

        # In a real application, you'd likely store this file and return a URL
        # For now, we'll return the path and expect the frontend to handle it.
        return {"bpm": bpm, "status": "audio_generation_successful", "audio_file_path": wav_file_path}

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
