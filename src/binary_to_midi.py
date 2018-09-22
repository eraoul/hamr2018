"""Convert form our numpy format (neural net output) to a midi file."""

from collections import namedtuple
import glob, os
from itertools import chain
import json
import numpy as np
import pickle

from music21 import converter, instrument, note, chord, stream
from music21.midi.translate import streamToMidiFile

TXT_TOKENIZED = '../Bach-Two_Part_Inventions_MIDI_Transposed/txt_tokenized'
CHUNK_SIZE = 4  # MEASURES
PADDING_TOKEN = 126



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
        if value == PADDING_TOKEN:
            continue
        if timestep % 2 == 0:
            # midi # case
            midi = value
        else:
            # duration case
            dur_string = index_to_symbol[value]
            notes.append(Note(midi, dur_string))

    return notes


PC_TO_NOTE = {0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'Bb', 11: 'B'}


def midinote_to_pc_octave(note):
    pc = note % 12
    octave = note // 12 - 1
    notename = PC_TO_NOTE[pc]
    return '%s%d' % (notename, octave)


def dur_string_to_quarterlength(dur_string):
    dur = eval(dur_string)
    return dur


def notelist_to_midi(notes, filename='test.mid'):
    s = stream.Stream() 

    for n in notes:
        midinote = note.Note(midinote_to_pc_octave(n.midi))
        midinote.quarterLength = dur_string_to_quarterlength(n.dur_string)
        s.append(midinote)

    mf = streamToMidiFile(s)
    mf.open(filename, 'wb')
    mf.write()
    mf.close()


def convert_array_to_midi(data, output_filename):
    """Use this function to convert a numpy 2D array (neural net output) to a MIDI file."""
    notes = binary_to_notelist(data)
    notelist_to_midi(notes, output_filename)



if __name__ == '__main__':
    notes = binary_to_notelist(np.array([[0,1,0,0,0,0,0,0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 
        0, .9, 0], [0, 0, 0.1, 0.2, 0, 0, 0, 0]]))
    print(notes)

    notelist_to_midi(notes)

    print('DONE!')



