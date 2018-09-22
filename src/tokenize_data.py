import numpy as np
import glob
import os

from keras.utils import to_categorical

os.environ["KERAS_BACKEND"] = "tensorflow"

PADDING_TOKEN = 126  # start token is 127; avoid conflict.


def pad_left(ex, pad_to_size):
    """ex is an example list of ints. Pad left to the given length."""
    pad_len = pad_to_size - len(ex)
    padded = [PADDING_TOKEN] * pad_len
    padded.extend(ex)
    return padded

def pad_right(ex, pad_to_size):
    """ex is an example list of ints. Pad left to the given length."""
    pad_len = pad_to_size - len(ex)
    padding = [PADDING_TOKEN] * pad_len
    ex.extend(padding)
    return ex

def add_example_to_list(filename, example_list, pad_to_size, pad_on_right=False):
    example = []
    with open(filename, "r") as f:
        content = f.read().strip().split(' ')
        for char in content:
            if not char == '':
                example.append(int(char))

    if pad_on_right:
        example_padded = pad_right(example, pad_to_size)
    else:
        example_padded = pad_left(example, pad_to_size)

    example_list.append(example_padded)


def tokenize_data(data_folder='../Bach-Two_Part_Inventions_MIDI_Transposed/txt_tokenized'):
    # Vectorize the data. Returns a tuple of inputs, outputs.
    input_texts = []
    target_texts = []

    # find max sequence length
    lens = []
    for file in glob.glob(os.path.join(data_folder, '*.txt')):
        print(file)
        with open(file) as f:
            content = f.read().strip().split(' ')
            lens.append(len(content))
    # max_length = max(lens)
    max_length = 250
    print('max length', max_length)

    for file in glob.glob(os.path.join(data_folder, '*.txt')):
        print(file)
        filename = os.path.basename(file)
        if filename.split("_")[3] == "0":
            add_example_to_list(file, input_texts, max_length, pad_on_right=False)
        elif filename.split("_")[3] == "1":
            add_example_to_list(file, target_texts, max_length, pad_on_right=True)

    input_examples = np.array(input_texts)
    target_examples = np.array(target_texts)

    input_arrays = np.array([to_categorical(ex, num_classes=128) for ex in input_examples])
    target_arrays = np.array([to_categorical(ex, num_classes=128) for ex in target_examples])

    return input_arrays, target_arrays
