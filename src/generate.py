import logging

import tensorflow as tf
from tensorflow.python.training.saver import Saver

from binary_to_midi import convert_array_to_midi
from config import *
from dataset_load import create_tfrecords_iterator
from models import seq2seq_generate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sequence(model_folder):
    itr = create_tfrecords_iterator(TEST_TFRECORDS, batch_size=1, shuffle_buffer=SHUFFLE_BUFFER)
    x, _, y = itr.get_next()
    in_seq = tf.unstack(tf.reshape(x, [1, VEC_LENGTH, TIMESTEPS]), axis=-1)
    out_seq = tf.unstack(tf.reshape(y, [1, VEC_LENGTH, TIMESTEPS]), axis=-1)
    model = seq2seq_generate

    gen_seq = model(in_seq)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = Saver()
        saver.restore(sess, os.path.join(model_folder, "model.ckpt"))
        for n in range(NUM_TEST_EXAMPLES):
            input_sequence, output_original, output_generated = sess.run([in_seq, out_seq, gen_seq])

            # Write to disk.
            output_folder = os.path.join(model_folder, 'generated_sequences')
            try:
                os.makedirs(output_folder)
            except FileExistsError:
                pass
            convert_array_to_midi(np.array([i[0] for i in input_sequence]), os.path.join(output_folder, '{}_input.mid'.format(n)))
            convert_array_to_midi(np.array([i[0] for i in output_generated]), os.path.join(output_folder, '{}_output.mid'.format(n)))
            convert_array_to_midi(np.array([i[0] for i in output_original]), os.path.join(output_folder, '{}_orig_cont.mid'.format(n)))


if __name__ == '__main__':
    print('-------------------------------------------------')
    print('HAMR 2018: Modeling antiphony with seq2seq models')
    print('-------------------------------------------------')

    model_folder = os.path.join('..', 'models', 's2s_2018-10-11_18-12-19')
    generate_sequence(model_folder)
