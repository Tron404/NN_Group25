import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pretty_midi
import sys
import collections
import seaborn as sns

# Convert a MIDI file to a data frame of notes
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

def getDuration(df: pd.DataFrame) -> np.double:
    return np.round(np.sum(df["duration"]), 2)

def createPlots(df: pd.DataFrame) -> None:
    directory_path = "graphs\\"
    palette = sns.color_palette("Set2")
    palette[0], palette[1] = palette[1], palette[0]
    sns.set(rc={'figure.figsize':(16,9)})
    sns.set_palette(palette)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=2)

    ####### Duration
    duration_density = sns.kdeplot(df["duration"], alpha=0.6, fill=True)
    duration_density.set(xlabel = "Duration of songs [s]", ylabel="Probability density of durations", title="Distribution of song duration across the data", xlim=(np.min(df["duration"])-100,np.max(df["duration"])+100))
    sns.despine()
    plt.savefig(directory_path + "duration.png", dpi=1000)

    ####### Tempo
    # tempo_dict = collections.defaultdict(int)
    # for t in df["tempo"]:
    #     tempo_dict[t] += 1
    # tempo_dict = sorted(tempo_dict.items(), reverse=False, key=lambda x:x[1])
    # tempo = [g[0] for g in tempo_dict]
    # tempo_freq = [f[1] for f in tempo_dict]
    
    # tempo_barplot = sns.barplot(x=tempo, y=tempo_freq)
    # tempo_barplot.set(xlabel = "Tempo of songs [bpm]", ylabel="Frequency of apparitions", title="Distribution of tempi across the data")
    # sns.despine()
    # plt.savefig(direction_path + "tempo.png", dpi=1000)

    ####### Sub-genre
    # genre_frequency = collections.defaultdict(int)
    # for song in df["sub-genre"]:
    #     genre_frequency[song] += 1
    # genre_frequency = sorted(genre_frequency.items(), reverse=True, key=lambda x:x[1])
    # genre = [g[0] for g in genre_frequency]
    # genre_freq = [f[1] for f in genre_frequency]

    # tempo_barplot = sns.barplot(x=genre, y=genre_freq)
    # tempo_barplot.set(xlabel = "Rock sub-genre", ylabel="Frequency of apparitions", title="Distribution of rock sub-genres across the data")
    # sns.despine()
    # plt.savefig(direction_path + "subgenre.png", dpi=1000)

    plt.show()

def main():
    file_path = "groove_safe"
    files = os.listdir(file_path)

    all_songs_features= []
    idx = 0
    for file in files[:500]:
        if file.endswith(".midi"):
            sys.stdout.write(f"File {idx}/{len(files)-5}" + "\r")
            notes= midi_to_notes(file_path + "\\"+ file)
            all_songs_features.append([re.split(r"_", file)[1], re.split(r"_", file)[2], getDuration(notes)]) # sub-genre, tempo, duration
            idx += 1

    print("")
    print("=============")
    all_songs_features = pd.DataFrame({"sub-genre": [song[0] for song in all_songs_features], "tempo": [int(song[1]) for song in all_songs_features], "duration": [song[2] for song in all_songs_features]})
    createPlots(all_songs_features)

main()