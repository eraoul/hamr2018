import numpy as np
import glob
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

data_path = 'Bach-Two_Part_Inventions_MIDI_Transposed/txt_tokenized'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()


for file in glob.glob("./*.txt"):
    if file.split("_")[3] == "0":
        with open(file, "r") as f:
            content = f.read().strip().split(' ')
            for char in content:
                #print(char)
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


input_cat = to_categorical(x, num_classes=128)
target_cat = to_categorical(y, num_classes=128)
