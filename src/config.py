""" CONFIG """
import os
import numpy as np

SIG_DIGITS = 4

NUM_MIDI_CLASSES = 128
NUM_DURATION_CLASSES = 24
VEC_LENGTH = max(NUM_MIDI_CLASSES, NUM_DURATION_CLASSES)

TIMESTEPS = 250
NUM_LSTM_NODES = 1024  # Num of intermediate LSTM nodes

BATCH_SIZE = 1  # DON'T CHANGE IT SO FAR!
NUM_EPOCHS = 100

LR = 0.01
DROPOUT = 0.3

TRAIN_FOLDER = os.path.join('..', 'data', 'training_set')
TEST_FOLDER = os.path.join('..', 'data', 'validation_set')
OUTPUT_FOLDER = os.path.join('..', 'generated_sequences')
MODEL_PATH = os.path.join('..', 's2s.h5')

NUM_TRAINING_EXAMPLES = len(os.listdir(TRAIN_FOLDER))
NUM_TEST_EXAMPLES = len(os.listdir(TEST_FOLDER))
NUM_STEPS_PER_EPOCH = int(np.ceil(NUM_TRAINING_EXAMPLES / BATCH_SIZE))
NUM_TEST_STEPS_PER_EPOCH = int(np.ceil(NUM_TEST_EXAMPLES / BATCH_SIZE))
