""" CONFIG """
import os
from datetime import datetime

import numpy as np

SIG_DIGITS = 4

# PATHS
TRAIN_FOLDER = os.path.join('..', 'data', 'training_set')
TEST_FOLDER = os.path.join('..', 'data', 'validation_set')
TRAIN_TFRECORDS = os.path.join('..', 'data', 'train.tfrecords')
TEST_TFRECORDS = os.path.join('..', 'data', 'validation.tfrecords')
# MODEL_PATH = os.path.join('..', 's2s.h5')
MODEL_FOLDER = os.path.join('..', 'models', 's2s_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
TXT_TOKENIZED = os.path.join('..', 'Bach-Two_Part_Inventions_MIDI_Transposed', 'txt_tokenized')

# MODEL CONFIGURATION
NUM_LSTM_NODES = 256  # Num of intermediate LSTM nodes
LR = 0.01
DROPOUT = 0.3

# DATA PROCESSING
NUM_MIDI_CLASSES = 128
NUM_DURATION_CLASSES = 24
VEC_LENGTH = max(NUM_MIDI_CLASSES, NUM_DURATION_CLASSES)
CHUNK_SIZE = 4  # MEASURES
PADDING_TOKEN = 126
START_TOKEN = 127
TIMESTEPS = 250

# TRAINING
BATCH_SIZE = 16  # DON'T CHANGE IT SO FAR!
SHUFFLE_BUFFER = 1000
NUM_EPOCHS = 100
NUM_TRAINING_EXAMPLES = len(os.listdir(TRAIN_FOLDER))
NUM_TEST_EXAMPLES = len(os.listdir(TEST_FOLDER))
NUM_STEPS_PER_EPOCH = int(np.ceil(NUM_TRAINING_EXAMPLES / BATCH_SIZE))
NUM_TEST_STEPS_PER_EPOCH = int(np.ceil(NUM_TEST_EXAMPLES / BATCH_SIZE))
NUM_STEPS = (NUM_STEPS_PER_EPOCH + 1) * NUM_EPOCHS

