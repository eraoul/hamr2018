import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.nn_ops import softmax

from config import NUM_LSTM_NODES, VEC_LENGTH, TIMESTEPS
from tokenize_data import create_start_token


def _seq2seq_define_cells():
    # Encoder section
    encoder_lstm_1 = rnn.BasicLSTMCell(NUM_LSTM_NODES, forget_bias=1.0, name='encoder_lstm_1')
    encoder_lstm_2 = rnn.BasicLSTMCell(NUM_LSTM_NODES, forget_bias=1.0, name='encoder_lstm_2')
    # Decoder section - set up the decoder using encoder_states as initial state.
    decoder_lstm_1 = rnn.BasicLSTMCell(NUM_LSTM_NODES, forget_bias=1.0, name='decoder_lstm_1')
    decoder_lstm_2 = rnn.BasicLSTMCell(NUM_LSTM_NODES, forget_bias=1.0, name='decoder_lstm_2')
    return decoder_lstm_1, decoder_lstm_2, encoder_lstm_1, encoder_lstm_2


def seq2seq_train(encoder_inputs, decoder_inputs):
    decoder_lstm_1, decoder_lstm_2, encoder_lstm_1, encoder_lstm_2 = _seq2seq_define_cells()

    encoder_outputs_1, encoder_states_1 = rnn.static_rnn(encoder_lstm_1, encoder_inputs, dtype=tf.float32)
    encoder_outputs_2, encoder_states_2 = rnn.static_rnn(encoder_lstm_2, encoder_outputs_1, dtype=tf.float32)

    decoder_outputs_1, _ = rnn.static_rnn(decoder_lstm_1, decoder_inputs, initial_state=encoder_states_1)
    decoder_outputs_2, _ = rnn.static_rnn(decoder_lstm_2, decoder_outputs_1, initial_state=encoder_states_2)

    # Final dense layer
    logits = []
    W = tf.Variable(tf.random_normal([NUM_LSTM_NODES, VEC_LENGTH], stddev=0.1), name="dense_weights")
    b = tf.Variable(tf.zeros([VEC_LENGTH]), name="dense_biases")
    for o in decoder_outputs_2:
        l = tf.add(tf.matmul(o, W), b)
        logits.append(l)
    logits = tf.transpose(tf.stack(logits), [1, 0, 2])
    return logits


def seq2seq_generate(encoder_inputs):
    decoder_lstm_1, decoder_lstm_2, encoder_lstm_1, encoder_lstm_2 = _seq2seq_define_cells()

    encoder_outputs_1, encoder_states_1 = rnn.static_rnn(encoder_lstm_1, encoder_inputs, dtype=tf.float32)
    encoder_outputs_2, encoder_states_2 = rnn.static_rnn(encoder_lstm_2, encoder_outputs_1, dtype=tf.float32)

    di = [tf.convert_to_tensor(create_start_token())]
    es1, es2 = encoder_states_1, encoder_states_2
    logits = []
    W = tf.Variable(tf.random_normal([NUM_LSTM_NODES, VEC_LENGTH], stddev=0.1), name="dense_weights")
    b = tf.Variable(tf.zeros([VEC_LENGTH]), name="dense_biases")
    for t in range(TIMESTEPS):
        o1, es1 = rnn.static_rnn(decoder_lstm_1, di, initial_state=es1)
        o2, es2 = rnn.static_rnn(decoder_lstm_2, o1, initial_state=es2)
        l = tf.add(tf.matmul(o2[0], W), b)
        logits.append(l)
        # if tf.cond(tf.equal(tf.argmax(l, axis=-1), PADDING_TOKEN)):
        #     break
        # else:
        #     di = [l]
        di = [l]
    generated_sequence = [softmax(l) for l in logits]

    return generated_sequence

