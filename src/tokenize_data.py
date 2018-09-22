import numpy as np
import glob
import os

from keras.utils import to_categorical

os.environ["KERAS_BACKEND"] = "tensorflow"

data_path = '../Bach-Two_Part_Inventions_MIDI_Transposed/txt_tokenized'


def tokenize_data(data_folder):
    # Vectorize the data. Returns a tuple of inputs, outputs.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    for file in sorted(glob.glob(os.path.join(data_folder, "*.txt"))):
        if file.split("_")[3] == "0":
            with open(file, "r") as f:
                content = f.read().strip().split(' ')
                for char in content:
                    # print(char)
                    if not char == '':
                        input_texts.append(char)
                        if char not in input_characters:
                            input_characters.add(char)
        elif file.split("_")[3] == "1":
            with open(file, "r") as f:
                content = f.read().strip().split(' ')
                for char in content:
                    if not char == '':
                        target_texts.append(char)
                        if char not in target_characters:
                            target_characters.add(char)

    input_texts = np.array([int(e) for e in input_texts])
    target_texts = np.array([int(e) for e in target_texts])

    input_cat = to_categorical(input_texts, num_classes=128)
    target_cat = to_categorical(target_texts, num_classes=128)

    return input_cat, target_cat
