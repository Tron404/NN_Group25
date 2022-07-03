### SOURCE: https://www.tensorflow.org/tutorials/audio/music_generation

import collections
import os
import glob
import numpy as np
import random
import pathlib
import pandas as pd
import pretty_midi
import tensorflow as tf
from keras import backend

training_sample_size = 500
note_baseline = 35.0 # init value, it will change after going through the data

################### Data preprocessing/extraction ###################
# After generating own notes, convert the extracted notes from a file to a MIDI file (this is not pre-processing, but post-processing)
def notes_to_midi(notes, output_file, instrument_name, velocity=100):
    #create an MIDI object
    pm = pretty_midi.PrettyMIDI()
    #create the instrument with the corresponding instrument number
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        #create a note starting at star and ending at end
        note = pretty_midi.Note(velocity = velocity, pitch = int(note['pitch']), start = start, end = end)
        #add the note to the instrument
        instrument.notes.append(note)
        #new prev_start is the end of the previous note
        prev_start = start

    #add the instrument to the MIDI object pm
    pm.instruments.append(instrument)
    #write out the MIDI data
    pm.write(output_file)
    return pm


# Convert a MIDI file to a data frame of notes ########Pre-processing
def midi_to_notes(midi_file):
    # make a MIDI object
    pm = pretty_midi.PrettyMIDI(midi_file)
    # decide what instrument it is
    instrument = pm.instruments[0]
    # make an empty list for the notes
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    # add all the relevant information to the notes
    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)   # key.append(value)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        notes['name'].append(pretty_midi.note_number_to_drum_name(note.pitch))
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()}) # one big dataframe of all notes of one file


################### Data set ###################
# Extract all the notes to train on ########Pre-processing
def extract_training_data(filenames):
    num_files = len(filenames)
    all_notes = []  # make a list containing all the dataframes
    global note_baseline

    for file in filenames:  # filenames[:num_files]:
        aux = midi_to_notes(file)
        note_baseline = np.min([note_baseline, np.min(aux["pitch"])])
        all_notes.append(aux)

    print("Number of songs from which notes were extracted:", len(all_notes))
    random_song_notes = random.choice(all_notes)
    all_notes = pd.concat(all_notes)  # concatinate all notes -> make one dataframe
    
    print("Number of notes parsed:", len(all_notes))    # number of notes in the concatinated dataframe
   
    key_order = ["pitch", "step", "duration"]
    sorted_notes = np.stack([all_notes[key] for key in key_order], axis=1)  # sort the notes over the pitch, step and duration like a stack of the notes
    notes_data_set = tf.data.Dataset.from_tensor_slices(sorted_notes) # make a data set of the stack

    return random_song_notes, notes_data_set, len(all_notes)


# Generates a sequence of notes to be fed to the network. A sequence will be split in a subsequence of input notes
# and a final note as a label (i.e., the note the NN will have to predict). ########Pre-processing
def create_sequences(dataset, seq_length, vocab_size):
    seq_length = seq_length + 1

    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True) # each window is a data set that contains a subset

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
    vocabulary_size = 61
    sequence_data_set = create_sequences(notes_data_set, sequence_length, vocabulary_size)

    batch_size = 64
    print(parsed_notes_total)
    buffer_size = parsed_notes_total - sequence_length  # the number of items in the dataset
    all_data = (sequence_data_set.shuffle(buffer_size).batch(batch_size, drop_remainder=True).cache().prefetch(
        tf.data.experimental.AUTOTUNE))

    training_data_set, testing_data_set = all_data.take(int(0.7*len(list(all_data)))), all_data.skip(int(0.7*len(list(all_data))))
    
    return sequence_length, vocabulary_size, training_data_set, testing_data_set

###### Pre-processing ends here!!! #######


################### Model creation + training ###################
def mse_with_positive_pressure(y_true, y_pred):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)

@tf.function
def custom_huber(y_true, y_pred):
    delta = tf.cast(1.1, dtype=backend.floatx())
    y_true_tf = tf.cast(y_true, dtype=backend.floatx())
    y_pred_tf = tf.cast(y_pred, dtype=backend.floatx())
    diff = tf.subtract(y_pred_tf, y_true_tf)
    abs_diff = tf.abs(diff)

    half = tf.convert_to_tensor(0.5, dtype=abs_diff.dtype)
    hbl = tf.where(abs_diff <= delta, half * tf.square(diff), delta * abs_diff - half * tf.square(delta))
    note_baseline_tf = tf.cast(note_baseline, dtype=abs_diff.dtype)

    pressure = tf.subtract(y_pred_tf, note_baseline_tf)
    pressure = tf.divide(pressure, note_baseline_tf)
    pressure = tf.multiply(pressure, tf.cast(5.0, dtype=pressure.dtype))
    pressure = tf.maximum(-pressure, 0.0)

    return backend.mean(hbl + pressure, axis=-1)

def build_model(sequence_length):
    learning_rate = 0.001

    inputs = tf.keras.Input((sequence_length, 3))
    lstm_layer_1 = tf.keras.layers.LSTM(61, return_sequences = True, kernel_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
    lstm_layer_2 = tf.keras.layers.LSTM(61, kernel_regularizer=tf.keras.regularizers.L2(0.01))(lstm_layer_1)

    outputs = {
        "pitch": tf.keras.layers.Dense(61, name="pitch")(lstm_layer_2),
        "step": tf.keras.layers.Dense(1, name="step")(lstm_layer_2),
        "duration": tf.keras.layers.Dense(1, name="duration")(lstm_layer_2),
    }

    loss = {
        "pitch": custom_huber,
        "step": mse_with_positive_pressure,
        "duration": mse_with_positive_pressure,
    }

    model = tf.keras.Model(inputs, outputs)

    model.compile(loss=loss,
                  loss_weights={'pitch': 0.5, 'step': 1.0, 'duration': 1.0},
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return model


################### Note generation ###################
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


def notes_generator(model, random_song_notes, sequence_length, vocabulary_size):
    temperature = 5.0 # adds a bit of randomness  to the generated notes
    num_predictions = 300
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
    out_pm = notes_to_midi(generated_notes, output_file="output_LSTM.mid", instrument_name="Melodic Tom")


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
    sequence_length, vocabulary_size, training_data_set, testing_data_set = oragnize_training_data_and_parameters(notes_data_set, notes_count)

    # Model building + training
    model = build_model(sequence_length)

    # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, save_weights_only=True),
    #              tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True), ]
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, save_weights_only=True), ]
    
    epochs = 50
    metrics = model.fit(training_data_set, epochs=epochs, callbacks=callbacks, validation_data=testing_data_set)
    df = pd.DataFrame(metrics.history).to_csv("metrics_LSTM")

    model.save("LSTM")
    model.evaluate(testing_data_set)

    # Music generation
    notes_generator(model, random_song_notes, sequence_length, vocabulary_size)
    model.summary()


main()
