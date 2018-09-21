import glob, os, numpy
from music21 import converter, instrument, note, chord
from itertools import chain


def get_notes(file):
    '''
    get a string of all the notes/rest and their durations for a given file
    '''
    notes = []
    midi = converter.parse(file)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts: # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            string = " "+str(element.pitch.name)+str(element.pitch.octave)+" "+str(element.duration.quarterLength)
            notes.append(string)
        if getattr(element,'isRest',None) and element.isRest:
            string = " "+'Rest' + " "+ str(element.duration.quarterLength)
            notes.append(string)
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return '' .join(chain.from_iterable(notes))

def convert_midi_to_txt(directory):
    for file in glob.glob(directory+"/*.mid"):
        string = get_notes(file)
        with open(file+".txt", "w") as f:
            f.write(string)
            f.close()
