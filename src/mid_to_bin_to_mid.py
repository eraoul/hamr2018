from binary_to_midi import convert_array_to_midi
from seq2seq.tokenize_data import tokenize_data
import numpy as np

input_cat, target_cat = tokenize_data()

print(input_cat.shape, target_cat.shape)

#print(input_cat, target_cat)

#convert back to midi

EX_NUM = 0


for row in target_cat[EX_NUM]:
    if np.argmax(row) == 126:
        continue



convert_array_to_midi(input_cat[EX_NUM], 'tmp_%d_part1.mid' % EX_NUM)
convert_array_to_midi(target_cat[EX_NUM], 'tmp_%d_part2.mid' % EX_NUM)



