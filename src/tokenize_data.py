import numpy as np
import glob
import os

import tensorflow as tf
from keras.utils import to_categorical

from config import PADDING_TOKEN, TIMESTEPS, TRAIN_TFRECORDS, VALIDATION_TFRECORDS, TRAIN_FOLDER, VALIDATION_FOLDER, VEC_LENGTH, \
    START_TOKEN, NUM_MIDI_CLASSES, NUM_DURATION_CLASSES


def create_start_token():
    """
    Return a start token that is [0... 0 1] with a total number of elements equal to the number of MIDI channels.
    The start token is chosen in this way because Bach inventions never use the last MIDI channel.

    :return:
    """
    start_token = np.zeros((1, VEC_LENGTH), dtype=np.float32)
    start_token[:, START_TOKEN] = 1
    start_token[:, NUM_MIDI_CLASSES] = 1
    return start_token


def pad_left(ex, pad_to_size, pad_value=PADDING_TOKEN):
    """ex is an example list of ints. Pad left to the given length."""
    pad_len = pad_to_size - len(ex)
    padded = [pad_value] * pad_len
    padded.extend(ex)
    return padded


def pad_right(ex, pad_to_size, pad_value=PADDING_TOKEN):
    """ex is an example list of ints. Pad right to the given length."""
    pad_len = pad_to_size - len(ex)
    padding = [pad_value] * pad_len
    ex.extend(padding)
    return ex


def add_example_to_lists(filename, pitch_list, duration_list, pad_to_size, pad_on_right=False):
    pitches, durations = [], []
    with open(filename, "r") as f:
        content = f.read().strip().split(' ')
        for n, char in enumerate(content):
            if char != '':
                if n % 2 == 0:
                    pitches.append(int(char))
                else:
                    durations.append(int(char))

    if pad_on_right:
        pitches_padded = pad_right(pitches, pad_to_size)
        durations_padded = pad_right(durations, pad_to_size, 0)
    else:
        pitches_padded = pad_left(pitches, pad_to_size)
        durations_padded = pad_left(durations, pad_to_size, 0)

    pitch_list.append(pitches_padded)
    duration_list.append(durations_padded)
    return


def tokenize_data(data_folder, max_length):
    # Vectorize the data. Returns a tuple of inputs, outputs.
    input_pitches, input_durations = [], []
    target_pitches, target_durations = [], []

    for file in glob.glob(os.path.join(data_folder, '*.txt')):
        # print(file)
        filename = os.path.basename(file)
        if filename.split("_")[3] == "0":  # 0 and 1 encode right hand and left hand, respectively
            add_example_to_lists(file, input_pitches, input_durations, max_length, pad_on_right=False)
        elif filename.split("_")[3] == "1":  # if it's the left hand
            add_example_to_lists(file, target_pitches, target_durations, max_length, pad_on_right=True)
        else:
            raise ValueError("File name not in the supported format")

    input_pitches = np.array([to_categorical(ex, num_classes=NUM_MIDI_CLASSES) for ex in input_pitches])
    input_durations = np.array([to_categorical(ex, num_classes=NUM_DURATION_CLASSES) for ex in input_durations])
    target_pitches = np.array([to_categorical(ex, num_classes=NUM_MIDI_CLASSES) for ex in target_pitches])
    target_durations = np.array([to_categorical(ex, num_classes=NUM_DURATION_CLASSES) for ex in target_durations])

    input_arrays = np.concatenate((input_pitches, input_durations), axis=-1)
    target_arrays = np.concatenate((target_pitches, target_durations), axis=-1)
    return input_arrays, target_arrays


def transform_into_tfrecord(data_folder, output_path):
    if os.path.isfile(output_path):
        raise PermissionError("The output file already exists. Exiting to prevent data loss.")
    inputs, targets = tokenize_data(data_folder, TIMESTEPS)
    start_token = create_start_token()

    with tf.python_io.TFRecordWriter(output_path) as writer:
        for i, t in zip(inputs, targets):
            di = np.concatenate((start_token, t[:-1, :]), axis=0)
            i = i.flatten()
            t = t.flatten()
            di = di.flatten()
            example = tf.train.Example()
            example.features.feature["input"].float_list.value.extend(i)
            example.features.feature["decoder_input"].float_list.value.extend(di)
            example.features.feature["target"].float_list.value.extend(t)
            writer.write(example.SerializeToString())
    return


if __name__ == '__main__':
    # tokenize_data(TXT_TOKENIZED, TIMESTEPS)
    transform_into_tfrecord(TRAIN_FOLDER, TRAIN_TFRECORDS)
    transform_into_tfrecord(VALIDATION_FOLDER, VALIDATION_TFRECORDS)
