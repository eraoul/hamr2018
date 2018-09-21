import glob, os, numpy
from music21 import converter, instrument, note, chord
from itertools import chain

import json
import pickle

PATH = 'Bach-Two_Part_Inventions_MIDI_Transposed/txt'
OUTPUT_PATH = 'Bach-Two_Part_Inventions_MIDI_Transposed/txt_tokenized'
CHUNK_SIZE = 4  # MEASURES


def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(filename):
    with open(filename) as f:
        return pickle.load(f)


def generate_duration_tokens(directory):
    symbols = set()
    for file in glob.glob(os.path.join(directory, '*.txt')):
        #print(file)
        with open(file) as f:
            for line in f:
                tokens = line.strip().split(' ')
                
                for i, token in enumerate(tokens):
                    # Skip MIDI or REST tokens
                    if i % 2 == 0:
                        continue

                    symbols.add(token)
    symbol_list = sorted(symbols)
    symbol_to_index = {s: idx for idx,s in enumerate(symbol_list)}
    index_to_symbol = {idx: s  for idx,s in enumerate(symbol_list)}

    return symbol_to_index, index_to_symbol


def simplify_text(directory, output_directory, symbol_to_index):
    for file in glob.glob(os.path.join(PATH, '*.txt')):
        #print(file)
        output_filename = os.path.join(OUTPUT_PATH, os.path.basename(file))
        with open(file) as f:
            with open(output_filename, 'w') as outfile:
                for line in f:
                    line_out = []
                    tokens = line.strip().split(' ')
                    for i, token in enumerate(tokens):
                        # Don't tokenize MIDI or REST tokens
                        if i % 2 == 0:
                            line_out.append(token)
                        else:
                            line_out.append(str(symbol_to_index[token]))
                    outfile.write(' '.join(line_out) + '\n')


if __name__ == '__main__':
    symbol_to_index, index_to_symbol = generate_duration_tokens(PATH)
    print (symbol_to_index)

    simplify_text(PATH, OUTPUT_PATH, symbol_to_index)


    write_pickle(os.path.join(OUTPUT_PATH, 'symbol_to_index.pkl'), symbol_to_index)

    print('DONE!')
