import logging
from functools import partial

import keras
from keras.layers import Input
from keras.models import Model

from binary_to_midi import convert_array_to_midi
from config import *
from utils import create_start_token, example_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load from disk; reconstruct inference model
# Returns a function that performs inference on a given sequence.
def seq2seq_from_models(encoder_model1, encoder_model2, decoder_model, input_seq):
    # Encode the input as state vectors.
    h1, c1 = encoder_model1.predict(input_seq)
    h2, c2 = encoder_model2.predict(input_seq)
    states_value1 = [h1, c1]
    states_value2 = [h2, c2]

    # Generate first input: Start vector.
    target_seq = create_start_token()

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    output_sequence = []
    for step in range(TIMESTEPS):
        z = decoder_model.predict([np.expand_dims(target_seq, 0)] + states_value1 + states_value2)
        out_vec, h1, c1, h2, c2 = z
        output_sequence.append(out_vec[0, 0, :])

        # Update the target sequence (of length 1).
        sampled = np.argmax(out_vec[0, 0, :])
        target_seq = np.zeros((1, VEC_LENGTH))
        target_seq[0, sampled] = 1

        # Update states
        states_value1 = [h1, c1]
        states_value2 = [h2, c2]

        if sampled == 126:
            break

    return np.array(output_sequence)


def load_model(model_path):
    logging.info('Loading model...')
    model = keras.models.load_model(model_path)

    logging.info('Reconstructing model architecture:')
    encoder_inputs = model.get_layer(name='encoder_input').input
    encoder_lstm_1 = model.get_layer(name='encoder_lstm_1')
    encoder_states1 = encoder_lstm_1.output[1:3]
    encoder_lstm_2 = model.get_layer(name='encoder_lstm_2')
    encoder_states2 = encoder_lstm_2.output[1:3]

    decoder_inputs = model.get_layer(name='decoder_lstm_1').input[0]

    decoder_lstm1 = model.get_layer(name='decoder_lstm_1')
    decoder_lstm2 = model.get_layer(name='decoder_lstm_2')
    decoder_dense = model.get_layer(name='decoder_output')

    # Run Model

    # Define sampling models
    encoder_model1 = Model(encoder_inputs, encoder_states1)
    encoder_model2 = Model(encoder_inputs, encoder_states2)
    encoder_model2.summary()

    decoder_state_input_h1 = Input(shape=(NUM_LSTM_NODES,), name='inference_decoder_h1')
    decoder_state_input_c1 = Input(shape=(NUM_LSTM_NODES,), name='inference_decoder_c1')
    decoder_state_input_h2 = Input(shape=(NUM_LSTM_NODES,), name='inference_decoder_h2')
    decoder_state_input_c2 = Input(shape=(NUM_LSTM_NODES,), name='inference_decoder_c2')
    decoder_states_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]
    decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]

    zz1 = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs1)
    decoder_outputs1new, decoder_state_h1, decoder_state_c1 = zz1

    zz2 = decoder_lstm2(decoder_outputs1new, initial_state=decoder_states_inputs2)
    decoder_outputs2new, decoder_state_h2, decoder_state_c2 = zz2

    decoder_states1 = [decoder_state_h1, decoder_state_c1]
    decoder_states2 = [decoder_state_h2, decoder_state_c2]
    decoder_outputs_final = decoder_dense(decoder_outputs2new)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs1 + decoder_states_inputs2,
        [decoder_outputs_final] + decoder_states1 + decoder_states2)

    decoder_model.summary()

    return partial(seq2seq_from_models, encoder_model1, encoder_model2, decoder_model)


if __name__ == '__main__':
    print('-------------------------------------------------')
    print('HAMR 2018: Modeling antiphony with seq2seq models')
    print('-------------------------------------------------')

    validation_generator = example_generator(False)
    seq2seq = load_model(MODEL_PATH)

    # Process each file.
    logger.info('Processing...')
    for i in range(NUM_TEST_EXAMPLES):
        if i % 10 == 0:
            logger.info("Predicted {} sequences so far".format(i))

        x, output_original = next(validation_generator)
        input_seq = x[0]

        # Predict
        output_seq = seq2seq(input_seq)
        print(output_seq)
        # Write to disk.
        convert_array_to_midi(input_seq[0], os.path.join(OUTPUT_FOLDER, '{}_input.mid'.format(i)))
        convert_array_to_midi(output_seq, os.path.join(OUTPUT_FOLDER, '{}_output.mid'.format(i)))
        convert_array_to_midi(output_original[0], os.path.join(OUTPUT_FOLDER, '{}_orig_cont.mid'.format(i)))
