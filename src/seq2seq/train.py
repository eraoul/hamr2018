#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from IPython.display import SVG
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot

from tokenize_data import tokenize_data

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

""" CONFIG """
logger.info("Configuring the model")
SIG_DIGITS = 4

NUM_MIDI_CLASSES = 128
NUM_DURATION_CLASSES = 24
VEC_LENGTH = max(NUM_MIDI_CLASSES, NUM_DURATION_CLASSES)

TIMESTEPS = 250
NUM_LSTM_NODES = 1028  # Num of intermediate LSTM nodes

BATCH_SIZE = 1  # DON'T CHANGE IT SO FAR!
NUM_EPOCHS = 100

LR = 0.01
DROPOUT = 0.3

TRAIN_FOLDER = os.path.join('..', '..', 'data', 'training_set')
TEST_FOLDER = os.path.join('..', '..', 'data', 'validation_set')
NUM_TRAINING_EXAMPLES = len(os.listdir(TRAIN_FOLDER))
NUM_TEST_EXAMPLES = len(os.listdir(TEST_FOLDER))
NUM_STEPS_PER_EPOCH = int(np.ceil(NUM_TRAINING_EXAMPLES / BATCH_SIZE))
NUM_TEST_STEPS_PER_EPOCH = int(np.ceil(NUM_TEST_EXAMPLES / BATCH_SIZE))

""" DATA GENERATION """
logger.info("Preparing data")


def example_generator(train=True):
    folder = TRAIN_FOLDER if train else TEST_FOLDER
    encoder_input, decoder_output = tokenize_data(folder)
    start_token = create_start_token()
    while True:
        for ei, do in zip(encoder_input, decoder_output):
            # the decoder input needs a start token as the first element and removes the last note from the end
            di = np.concatenate((start_token, do[:-1, :]), axis=0)
            yield ([np.expand_dims(ei, 0), np.expand_dims(di, 0)], np.expand_dims(do, 0))


def create_start_token():
    """
    Return a start token that is [0... 0 1] with a total number of elements equal to the number of MIDI channels.
    The start token is chosen in this way because Bach inventions never use the last MIDI channel.

    :return:
    """
    start_token = np.zeros((1, NUM_MIDI_CLASSES))
    start_token[0, -1] = 1
    return start_token


training_generator = example_generator()
validation_generator = example_generator(False)
next(training_generator)

""" Model generation """
# See: https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
logger.info("Creating the model")

# Encoder section
encoder_lstm1 = LSTM(NUM_LSTM_NODES, return_sequences=True, return_state=True, name='encoder_lstm_1')
encoder_lstm2 = LSTM(NUM_LSTM_NODES, return_sequences=False, return_state=True, name='encoder_lstm_2')

encoder_inputs = Input(shape=(TIMESTEPS, VEC_LENGTH), name='encoder_input')
encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_inputs)  # instantiate the first encoder layer
encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)  # instantiate the second encoder layer

# Discard `encoder_outputs` and only keep the states.
encoder_states1 = [state_h1, state_c1]
encoder_states2 = [state_h2, state_c2]

# Decoder section
# Set up the decoder, using encoder_states as initial state.
decoder_lstm1 = LSTM(NUM_LSTM_NODES, return_sequences=True, return_state=True, name='decoder_lstm_1')
decoder_lstm2 = LSTM(NUM_LSTM_NODES, return_sequences=True, return_state=True, name='decoder_lstm_2')
decoder_dense = Dense(VEC_LENGTH, activation='softmax', name='decoder_output')

decoder_inputs = Input(shape=(None, VEC_LENGTH), name='decoder_input')
decoder_outputs1, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states1)
decoder_outputs2, _, _ = decoder_lstm2(decoder_outputs1, initial_state=encoder_states2)
decoder_outputs = decoder_dense(decoder_outputs2)

# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], [decoder_outputs])
logger.info("Summarizing the model")
model.summary()

""" Train """
optimizer = Adam(lr=.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto',
                                cooldown=0, min_lr=0)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

# Run training
logger.info("start the training")
model.fit_generator(training_generator, steps_per_epoch=NUM_STEPS_PER_EPOCH / 50,
                    validation_data=validation_generator,
                    validation_steps=NUM_TEST_STEPS_PER_EPOCH / 50,
                    verbose=1,
                    workers=1,
                    use_multiprocessing=False,
                    epochs=NUM_EPOCHS,
                    callbacks=[lr_callback, early_stopping_callback])

# Save model
model.save('s2s.h5')

# """ Run Model """
# # Define sampling models
# encoder_model1 = Model(encoder_inputs, encoder_states1)
# encoder_model2 = Model(encoder_inputs, encoder_states2)
#
# encoder_model1.summary()
# SVG(model_to_dot(encoder_model1, show_shapes=True).create(prog='dot', format='svg'))
#
# encoder_model2.summary()
# SVG(model_to_dot(encoder_model2, show_shapes=True).create(prog='dot', format='svg'))
#
# decoder_state_input_h1 = Input(shape=(NUM_LSTM_NODES,), name='inference_decoder_h1')
# decoder_state_input_c1 = Input(shape=(NUM_LSTM_NODES,), name='inference_decoder_c1')
# decoder_state_input_h2 = Input(shape=(NUM_LSTM_NODES,), name='inference_decoder_h2')
# decoder_state_input_c2 = Input(shape=(NUM_LSTM_NODES,), name='inference_decoder_c2')
# decoder_states_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]
# decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]
#
# zz1 = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs1)
# decoder_outputs1new, decoder_state_h1, decoder_state_c1 = zz1
#
# zz2 = decoder_lstm2(decoder_outputs1new, initial_state=decoder_states_inputs2)
# decoder_outputs2new, decoder_state_h2, decoder_state_c2 = zz2
#
# decoder_states1 = [decoder_state_h1, decoder_state_c1]
# decoder_states2 = [decoder_state_h2, decoder_state_c2]
# decoder_outputs_final = decoder_dense(decoder_outputs2new)
# decoder_model = Model([decoder_inputs] + decoder_states_inputs1 + decoder_states_inputs2,
#                       [decoder_outputs_final] + decoder_states1 + decoder_states2)
# decoder_model.summary()
# SVG(model_to_dot(decoder_model, show_shapes=True).create(prog='dot', format='svg'))
#
#
# def seq2seq(input_seq):
#     # Encode the input as state vectors.
#     h1, c1 = encoder_model1.predict(input_seq)
#     states_value1 = [h1, c1]
#     h2, c2 = encoder_model2.predict(input_seq)
#     states_value2 = [h2, c2]
#
#     # Generate first input: Start vector.
#     target_seq = np.zeros((1, VEC_LENGTH))
#     target_seq[0, 0] = 1  # first element is "1" to indicate "start"
#
#     # Sampling loop for a batch of sequences
#     # (to simplify, here we assume a batch of size 1).
#     stop_condition = False
#     output_sequence = []
#     step = 0
#     while step < OUTPUT_TIMESTEPS:
#         z = decoder_model.predict([np.expand_dims(target_seq, 0)] + states_value1 + states_value2)
#         out_vec, h1, c1, h2, c2 = z
#
#         sampled_output = np.argmax(out_vec[0, 0, :])
#
#         output_sequence.append(sampled_output)
#         step += 1
#
#         # Exit condition: either hit max length
#         # or find stop character.
#         # if (sampled_word == '</S>' or step > max_output_seq_len):
#         #    stop_condition = True
#
#         # Update the target sequence (of length 1).
#
#         target_seq = np.zeros((1, VEC_LENGTH))
#         target_seq[0, sampled_output] = 1
#
#         # Update states
#         states_value1 = [h1, c1]
#         states_value2 = [h2, c2]
#
#     return output_sequence
#
# [x, y], [target1, target2, target3] = next(training_generator)
#
# # In[32]:
#
#
# if False:
#     train = False
#     files = TRAIN_FILES if train else TEST_FILES
#
#     sum = 0
#     for i, file in enumerate(files):
#         x, y = joblib.load(file)
#         if i % 10 == 0:
#             print(i)
#         for yy in y:
#             print(yy.shape[0])
#             sum += yy.shape[0]
#     sum
#
# # # Convert Output to CSV and MIDI
#
# # In[33]:
#
#
# QUANTIZATION = 12  # smallest unit is 1/12 of a beat
# MAX_EVENT_BEATS = 4
#
# SIG_FIGS = 5
#
# MIDI_MIN = 21
# MIDI_MAX = 108
#
# MAX_EVENT_SUBBEATS = QUANTIZATION * MAX_EVENT_BEATS
#
# # In[34]:
#
#
# pc_to_degree_flat_key = [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6]
# pc_to_degree_sharp_key = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]
#
#
# def midi_to_mnn(midi, flat_key=True):
#     octave = 3 + midi // 12
#     pc = midi % 12
#     pc_to_degree = pc_to_degree_flat_key if flat_key else pc_to_degree_sharp_key
#     degree = pc_to_degree[pc]
#     return octave * 7 + degree + 4
#
#
# # In[35]:
#
#
# def seq_to_tuples(seq, start_time=100, channel=0, flat_key=True):
#     t = start_time  # time in beats, integer
#     subbeat = 0  # curent subbeat in beat for t, range is 0 to QUANTIZATION-1
#
#     notes = []
#
#     cur_note_start = 0
#     cur_note = None
#     cur_dur = None
#
#     for command, midi, dur in seq:
#         mnn = midi_to_mnn(midi, flat_key)
#         # Time-shift
#         if command == 3:
#             # Record note/rest start data.
#             cur_dur = dur
#             cur_note_start = round(t + subbeat / QUANTIZATION, SIG_FIGS)
#
#             # Update current time.
#             subbeat += dur
#             if subbeat > QUANTIZATION:
#                 subbeat = dur % QUANTIZATION
#                 t += dur // QUANTIZATION + 1
#
#         # Note on.
#         elif command == 2:
#             if cur_note:
#                 notes.append((cur_note_start, midi, mnn,
#                               round(cur_dur / QUANTIZATION, SIG_FIGS), channel))
#             cur_note = midi + MIDI_MIN - 1  # -1 for the 0 case
#
#         # Note off.
#         elif command == 1:
#             if cur_note:
#                 notes.append((cur_note_start, midi, mnn,
#                               round(cur_dur / QUANTIZATION, SIG_FIGS), channel))
#             cur_note = 0
#     return notes
#
#
# # In[36]:
#
#
# seq = seq2seq(np.expand_dims(x[1], axis=0))
#
# # In[37]:
#
#
# seq_to_tuples(seq)
#
#
# # In[38]:
#
#
# def print_tuples(tuples):
#     for t in tuples:
#         print(t)
#
#
# # In[39]:
#
#
# [x, y], [target1, target2, target3] = next(training_generator)
#
#
# # In[41]:
#
#
# def target_outputs_to_seq(N, target1, target2, target3):
#     return list(zip(np.argmax(target1[N], axis=1),
#                     np.argmax(target2[N], axis=1),
#                     np.argmax(target3[N], axis=1)))
#
#
# # In[42]:
#
#
# def target_outputs_to_tuples(N, target1, target2, target3):
#     return seq_to_tuples(target_outputs_to_seq(N, target1, target2, target3))
#
#
# # In[43]:
#
#
# def seq_to_csv(seq, filename='tst.csv'):
#     with open(filename, 'w') as f:
#         f.writelines(','.join(str(x) for x in tup) + '\n' for tup in seq_to_tuples(seq))
#
#
# # In[44]:
#
#
# for i in range(2):
#     seq = seq2seq(np.expand_dims(x[i], axis=0))
#     print('Output:')
#     print_tuples(seq_to_tuples(seq))
#     target_tuples = target_outputs_to_tuples(i, target1, target2, target3)
#     print()
#     print('prev should have been (target):')
#     print_tuples(target_tuples)
#     print('===============\n')
#
# # In[46]:
#
#
# seq_to_csv(seq)
#
# # In[ ]:
