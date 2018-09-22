from shutil import copyfile

import numpy as np
import os

tokens_path = os.path.join('..', 'Bach-Two_Part_Inventions_MIDI_Transposed', 'txt_tokenized')
# tokens = [os.path.join(tokens_path, fp) for fp in os.listdir(tokens_path)]
tokens = os.listdir(tokens_path)
output_folder = os.path.join('..', 'data')
training_folder = os.path.join(output_folder, 'training_set')
validation_folder = os.path.join(output_folder, 'validation_set')

np.random.seed(18)
for t in tokens:
    if t[-3:] == 'pkl':
        continue
    part = t.split('_')[3]
    if part == '1':
        continue
    t2 = t.split('_')
    t2[3] = '1'
    t2 = '_'.join(t2)
    if np.random.random() < 0.8:
        copyfile(os.path.join(tokens_path, t), os.path.join(training_folder, t))
        copyfile(os.path.join(tokens_path, t2), os.path.join(training_folder, t2))
    else:
        copyfile(os.path.join(tokens_path, t), os.path.join(validation_folder, t))
        copyfile(os.path.join(tokens_path, t2), os.path.join(validation_folder, t2))
