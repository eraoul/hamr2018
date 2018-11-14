"""Convert form our numpy format (neural net output) to a midi file."""

from collections import namedtuple
import os
import numpy as np
import pickle

from music21 import note, stream
from music21.midi.translate import streamToMidiFile

from config import TXT_TOKENIZED, PADDING_TOKEN, NUM_MIDI_CLASSES
from tokenize_data import pad_right


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def binary_to_notelist(pitches, durations):
    """pitches and durations are numpy arrays: [timestep, feature]"""

    assert len(pitches.shape) == 2
    assert len(durations.shape) == 2

    # Read the duration symbol table
    symbol_to_index = read_pickle(os.path.join(TXT_TOKENIZED, 'symbol_to_index.pkl'))
    index_to_symbol = {idx: symbol for idx, symbol in enumerate(symbol_to_index)}

    notes = []
    Note = namedtuple('Note', ['midi', 'dur_string'])

    left_pad = True
    for timestep in range(len(pitches)):
        pitch = np.argmax(pitches[timestep])
        duration = np.argmax(durations[timestep])
        print(timestep, pitch, duration, left_pad)
        if pitch == PADDING_TOKEN:
            if left_pad:
                continue
            else:
                break
        left_pad = False
        try:
            dur_string = index_to_symbol[duration]
        except KeyError:
            dur_string = None
        notes.append(Note(pitch, dur_string))

    return notes


PC_TO_NOTE = {0: 'C', 1: 'C#', 2: 'D', 3: 'E-', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'B-', 11: 'B'}


def midinote_to_pc_octave(note):
    pc = note % 12
    # TODO: The next line is a terrible hack! I'm ashamed of myself. Correct it!
    # It prevents the system from using non-existing octaves by misinterpreting the results of the network.
    octave = max(note // 12 - 1, 0)
    notename = PC_TO_NOTE[pc]
    return '%s%d' % (notename, octave)


def dur_string_to_quarterlength(dur_string):
    dur = eval(dur_string)
    return dur


def notelist_to_midi(notes, filename='test.mid'):
    s = stream.Stream()

    for n in notes:
        if n.midi == 0:
            midinote = note.Rest()
            print('REST!!')
        else:
            print('NOTE: midi=', n.midi)
            midinote = note.Note(midinote_to_pc_octave(n.midi))
        midinote.quarterLength = dur_string_to_quarterlength(n.dur_string)
        s.append(midinote)

    mf = streamToMidiFile(s)
    mf.open(filename, 'wb')
    mf.write()
    mf.close()


def convert_array_to_midi(data, output_filename):
    """Use this function to convert a numpy 2D array (neural net output) to a MIDI file."""
    pitches, durations = data[:, :NUM_MIDI_CLASSES], data[:, NUM_MIDI_CLASSES:]
    notes = binary_to_notelist(pitches, durations)
    notelist_to_midi(notes, output_filename)


if __name__ == '__main__':
    notes = binary_to_notelist(np.array([[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0,
                                                                                              0, .9, 0],
                                         [0, 0, 0.1, 0.2, 0, 0, 0, 0]]))
    print(notes)

    notelist_to_midi(notes)

    print('DONE!')
