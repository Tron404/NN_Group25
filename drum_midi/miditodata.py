### SOURCE: https://www.tensorflow.org/tutorials/audio/music_generation

import collections
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi

data_dir = pathlib.Path('data/groove-v1.0.0-midionly/groove')

filenames = glob.glob(str(data_dir/'*.mid*'))
print('Number of files:', len(filenames))

for file in filenames[:]:

    print(file)

    pm = pretty_midi.PrettyMIDI(file)

    print('Number of instruments:', len(pm.instruments))
    instrument = pm.instruments[0]
    if instrument.is_drum:
        print('Instrument name: drums')

    def midi_to_notes(midi_file: str) -> pd.DataFrame:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            notes['name'].append(pretty_midi.note_number_to_drum_name(note.pitch))
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

    raw_notes = midi_to_notes(file)
    print(raw_notes)
