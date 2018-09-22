"""Convert form our numpy format (neural net output) to a midi file."""

import glob, os, numpy
from music21 import converter, instrument, note, chord
from itertools import chain
from collections import namedtuple
import numpy as np
import json
import pickle

PATH = 'Bach-Two_Part_Inventions_MIDI_Transposed/txt'
TXT_TOKENIZED = 'Bach-Two_Part_Inventions_MIDI_Transposed/txt_tokenized'
CHUNK_SIZE = 4  # MEASURES


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def binary_to_notelist(data):
    """data is a numpy array: [timestep, feature]. Timestep 0 is a midi #, timestep 1 is a duration, timestep 2 is midi, etc..."""

    assert len(data.shape) == 2

    # Read the duration symbol table
    symbol_to_index = read_pickle(os.path.join(TXT_TOKENIZED, 'symbol_to_index.pkl'))    
    index_to_symbol = {idx: symbol for idx, symbol in enumerate(symbol_to_index)}

    notes = []
    Note = namedtuple('Note', ['midi', 'dur_string'])

    midi = None
    dur_string = None
    for timestep in range(len(data)):
        value = np.argmax(data[timestep])
        if timestep % 2 == 0:
            # midi # case
            midi = value
        else:
            dur_string = index_to_symbol[value]
            notes.append(Note(midi, dur_string))

    return notes

def notelist_to_midi(notes):
    pass


if __name__ == '__main__':
    notes = binary_to_notelist(np.array([[0,1,0,0,0,0,0,0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, .9, 0], [0, 0, 0.1, 0.2, 0, 0, 0, 0]]))
    print(notes)
    print('DONE!')



