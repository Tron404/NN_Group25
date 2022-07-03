import collections
from msilib.schema import Directory
import os
import glob
import shutil
import numpy as np
import random
import pathlib
import pandas as pd
import pretty_midi
import tensorflow as tf
import keras

from keras import backend

training_sample_size = 50
note_baseline = 35.0
model_name = input("GRU or LSTM?\n").upper()
number_songs = 1

file_path = model_name + "_songs/"
if os.path.exists(file_path):
    shutil.rmtree(file_path)
os.mkdir(file_path)

################### Data preprocessing/extraction ###################
# Convert the extracted notes from a file to a MIDI file
def notes_to_midi(notes, output_file, instrument_name, velocity=100):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(velocity = velocity, pitch = int(note['pitch']), start = start, end = end)
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(output_file)
    return pm

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


################### Data set ###################
# Extract all the notes to train on
def extract_training_data(filenames):
    # num_files = len(filenames)
    all_notes = []
 
    for file in filenames:  # filenames[:num_files]:
        all_notes.append(midi_to_notes(file))
    print("Number of songs from which notes were extracted:", len(all_notes))

    random_song_notes = random.choice(all_notes)
    all_notes = pd.concat(all_notes)

    print("Number of notes parsed:", len(all_notes))
    key_order = ["pitch", "step", "duration"]

    sorted_notes = np.stack([all_notes[key] for key in key_order], axis=1)
    notes_data_set = tf.data.Dataset.from_tensor_slices(sorted_notes)

    return random_song_notes, notes_data_set, len(all_notes)


# Generates a sequence of notes to be fed to the network. A sequence will be split in a subsequence of input notes
# and a final note as a label (i.e., the note the NN will have to predict).
def create_sequences(dataset, seq_length, vocab_size):
    seq_length = seq_length + 1

    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

    # Flatten the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        key_order = ["pitch", "step", "duration"]
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def oragnize_training_data_and_parameters(notes_data_set, parsed_notes_total):
    sequence_length = 50
    vocabulary_size = 60
    sequence_data_set = create_sequences(notes_data_set, sequence_length, vocabulary_size)

    batch_size = 64
    buffer_size = parsed_notes_total - sequence_length  # the number of items in the dataset
    training_data_set = (sequence_data_set.shuffle(buffer_size).batch(batch_size, drop_remainder=True).cache().prefetch(
        tf.data.experimental.AUTOTUNE))

    return sequence_length, vocabulary_size, training_data_set

def one_note_predictor(notes, model, temperature=1.0):
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions["pitch"]
    step = predictions["step"]
    duration = predictions["duration"]

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # select only non-negative step and duration
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


def notes_generator(model, random_song_notes, sequence_length, vocabulary_size, idx):
    print("===================================")
    print(f"Generating song {idx}\n")
    # temperature = 2.0
    temperature = 0.005 # adds a bit of randomness  to the generated notes
    num_predictions = 250
    key_order = ["pitch", "step", "duration"]
   
    sample_notes = np.stack([random_song_notes[key] for key in key_order], axis=1)

    input_notes = (sample_notes[:sequence_length] / np.array([vocabulary_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = one_note_predictor(input_notes, model, temperature)
        while pitch < note_baseline:
            pitch, step, duration = one_note_predictor(input_notes, model, temperature)

        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, "start", "end"))
    out_pm = notes_to_midi(generated_notes, output_file=f"{file_path}output_{idx}.mid", instrument_name="Melodic Tom")

def main():
# Initiation
    data_dir = pathlib.Path('groove_safe')
    checkpoints_path = "training_checkpoints/checkpoint_{epoch:04d}.ckpt"
    checkpoints_dir = os.path.dirname(checkpoints_path)
    filenames = glob.glob(str(data_dir / '*.mid*'))
    print('Number of files (songs):', len(filenames))
    print("===================")
    print("Extracting data")
    random_song_notes, notes_data_set, notes_count = extract_training_data(filenames[:training_sample_size])
    sequence_length, vocabulary_size, training_data_set = oragnize_training_data_and_parameters(notes_data_set, notes_count)

    # Model building + training
    model = tf.keras.models.load_model(model_name, compile=False)

    # Music generation
    for idx in range(number_songs):
        notes_generator(model, random_song_notes, sequence_length, vocabulary_size, idx+1)

main()
