import glob, os, numpy
from music21 import converter, instrument, note, chord
from itertools import chain



PATH = 'Bach-Two_Part_Inventions_MIDI_Transposed'
CHUNK_SIZE = 4  # MEASURES

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


def convert_parts(file):
    piece = converter.parse(file)

    # Find the max measure numbers.
    measure_lens = []
    for part in piece.parts:
        measures = part.makeMeasures()
        measure_lens.append(len(measures))

    max_measure = max(measure_lens) // CHUNK_SIZE * CHUNK_SIZE  # drop extra measures at the end

    chunks = [(i, i + CHUNK_SIZE) for i in range(0, max_measure, 4)]

    for i, part in enumerate(piece.parts):
        part_tuples=[]
        # try:
        #     track_name = part[0].bestName()
        # except AttributeError:
        #     track_name = 'None'
        track_name = file + 'part_%d' % i

    # num_parts = len(piece.parts)
    # counter=0
    # for part in piece.parts:
    #     counter += 1
    #     part_tuples=[]
    #     try:
    #         track_name = part[0].bestName()
    #     except AttributeError:
    #         track_name = 'part'+str(counter)


        #part_tuples.append(track_name)


        # Make measures and go through measures
        measures = part.makeMeasures()
        for start, end in chunks:
            part_tuples = []
            track_name = file + 'part_%d' % i
            for measure in measures[start: end]:
                for event in measure:
                    if getattr(event,'isNote',None) and event.isNote:
                        string = " "+str(event.pitch.name)+str(event.pitch.octave)+" "+str(event.quarterLength)
                        part_tuples.append(string)
                    if getattr(event,'isRest',None) and event.isRest:
                        string = " " + 'R' + " " + str(event.quarterLength)
                        part_tuples.append(string)
            track_name += '_%d-%d' % (start, end)
            part_tuples = '' .join(chain.from_iterable(part_tuples))
            with open(track_name + ".txt", "w") as f:
                f.write(str(part_tuples))
                f.close()


def convert_midi_to_txt(directory):
    for file in glob.glob(directory+"/*.mid"):
        string = get_notes(file)
        with open(file+".txt", "w") as f:
            f.write(string)
            f.close()

def convert_midi_to_txt_chunks(directory):
    for file in glob.glob(os.path.join(PATH, '*.mid')):
        string = convert_parts(file)



# def get_piece_chunks(filename):
#     print(file)

#if not os.path.exists(directory+"txt"):
#    os.makedirs("txt")


if __name__ == '__main__':
    convert_midi_to_txt_chunks(PATH)
    # for file in glob.glob(os.path.join(PATH, '*.mid')):
    #     get_piece_chunks(file)
