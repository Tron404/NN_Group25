import pretty_midi
import pandas as pd
import collections
import numpy as np
import os

def midi_to_notes(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list) # create a dictionary, in which the values are lists
    # functionally, if a key doesn't exist, defaultdict will create it instead of throwing an eror

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

    # create a df based on the notes dictionary; each key is a column
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

file_our = "output.mid"
file_train = "groove_safe\\1_rock-prog_125_beat_4-4_1.midi"
files = os.listdir("groove_safe")
g_max = -1
g_min = 130
for file in files:
    if file.endswith(".midi"):
        path = "groove_safe\\"
        df = midi_to_notes(path + file)
        # print(np.max(df["pitch"]), np.min(df["pitch"]))
        g_max = np.max([np.max(df["pitch"]), g_max])
        g_min = np.min([np.min(df["pitch"]), g_min])

print(f"global max: {g_max} - global min: {g_min}")