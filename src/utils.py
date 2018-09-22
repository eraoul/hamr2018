import numpy as np

from config import TRAIN_FOLDER, TEST_FOLDER, VEC_LENGTH
from tokenize_data import tokenize_data


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
    start_token = np.zeros((1, VEC_LENGTH))
    start_token[0, -1] = 1
    return start_token
