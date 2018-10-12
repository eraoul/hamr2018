import numpy as np
import glob
import os

import tensorflow as tf
from keras.utils import to_categorical

from config import PADDING_TOKEN, TIMESTEPS, TRAIN_TFRECORDS, TEST_TFRECORDS, TRAIN_FOLDER, TEST_FOLDER, VEC_LENGTH, \
    START_TOKEN


def create_start_token():
    """
    Return a start token that is [0... 0 1] with a total number of elements equal to the number of MIDI channels.
    The start token is chosen in this way because Bach inventions never use the last MIDI channel.

    :return:
    """
    start_token = np.zeros((1, VEC_LENGTH), dtype=np.float32)
    start_token[:, START_TOKEN] = 1
    return start_token


def pad_left(ex, pad_to_size):
    """ex is an example list of ints. Pad left to the given length."""
    pad_len = pad_to_size - len(ex)
    padded = [PADDING_TOKEN] * pad_len
    padded.extend(ex)
    return padded


def pad_right(ex, pad_to_size):
    """ex is an example list of ints. Pad right to the given length."""
    pad_len = pad_to_size - len(ex)
    padding = [PADDING_TOKEN] * pad_len
    ex.extend(padding)
    return ex


def add_example_to_list(filename, example_list, pad_to_size, pad_on_right=False):
    example = []
    with open(filename, "r") as f:
        content = f.read().strip().split(' ')
        for char in content:
            if char != '':
                example.append(int(char))

    if pad_on_right:
        example_padded = pad_right(example, pad_to_size)
    else:
        example_padded = pad_left(example, pad_to_size)

    example_list.append(example_padded)


def tokenize_data(data_folder, max_length):
    # Vectorize the data. Returns a tuple of inputs, outputs.
    input_texts = []
    target_texts = []

    # # find max sequence length
    # lens = []
    # for file in glob.glob(os.path.join(data_folder, '*.txt')):
    #     # print(file)
    #     with open(file) as f:
    #         content = f.read().strip().split(' ')
    #         lens.append(len(content))
    # # max_length = max(lens)
    # max_length = 250
    # # print('max length', max_length)

    for file in glob.glob(os.path.join(data_folder, '*.txt')):
        # print(file)
        filename = os.path.basename(file)
        if filename.split("_")[3] == "0":  # 0 and 1 encode right hand and left hand, respectively
            add_example_to_list(file, input_texts, max_length, pad_on_right=False)
        elif filename.split("_")[3] == "1":  # if it's the left hand
            add_example_to_list(file, target_texts, max_length, pad_on_right=True)
        else:
            raise ValueError("File name not in the supported format")

    input_examples = np.array(input_texts)
    target_examples = np.array(target_texts)

    input_arrays = np.array([to_categorical(ex, num_classes=128) for ex in input_examples])
    target_arrays = np.array([to_categorical(ex, num_classes=128) for ex in target_examples])

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
    transform_into_tfrecord(TEST_FOLDER, TEST_TFRECORDS)
