#!/usr/bin/env python
# coding: utf-8

import logging

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam

from config import *
from utils import example_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


""" DATA GENERATION """
logger.info("Preparing data")
training_generator = example_generator()
validation_generator = example_generator(False)

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
model.save(MODEL_PATH)
