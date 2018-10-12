#!/usr/bin/env python
# coding: utf-8

import logging

import tensorflow as tf
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler.model_analyzer import Profiler
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.training.saver import Saver

from config import *
from dataset_load import create_tfrecords_iterator
from models import seq2seq_train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

""" DATA GENERATION """
logger.info("Preparing data")
# training_generator = example_generator()
# validation_generator = example_generator(False)
# encoder_inputs = Input(shape=(TIMESTEPS, VEC_LENGTH), name='encoder_input')
# decoder_inputs = Input(shape=(None, VEC_LENGTH), name='decoder_input')
trn_itr = create_tfrecords_iterator(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)
vld_itr = create_tfrecords_iterator(TEST_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

handle = tf.placeholder(tf.string, shape=[])
x, di, y = tf.data.Iterator.from_string_handle(handle, trn_itr.output_types, trn_itr.output_shapes).get_next()
encoder_inputs = tf.unstack(tf.reshape(x, [BATCH_SIZE, VEC_LENGTH, TIMESTEPS]), axis=-1)
decoder_inputs = tf.unstack(tf.reshape(di, [BATCH_SIZE, VEC_LENGTH, TIMESTEPS]), axis=-1)
targets = tf.unstack(tf.reshape(y, [BATCH_SIZE, VEC_LENGTH, TIMESTEPS]), axis=-1)


""" Model generation """
# See: https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
logger.info("Creating the model")

logits = seq2seq_train(encoder_inputs, decoder_inputs)

""" Train """
loss = tf.losses.softmax_cross_entropy(targets, logits)
train_step = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.create_global_step())
tf.summary.scalar('loss', loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater(logits, 0), tf.cast(targets, tf.bool)), tf.float32))
tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    th = sess.run(trn_itr.string_handle())
    vh = sess.run(vld_itr.string_handle())

    merged = tf.summary.merge_all()
    trn_writer = FileWriter(os.path.join(MODEL_FOLDER, 'train'), sess.graph)
    vld_writer = FileWriter(os.path.join(MODEL_FOLDER, 'validation'))
    saver = Saver()
    profiler = Profiler(sess.graph)
    opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.trainable_variables_parameter())
            .with_file_output(os.path.join(MODEL_FOLDER, 'profile_model.txt')).build())
    profiler.profile_name_scope(options=opts)

    value_lv = None
    lv = tf.Summary()
    lv.value.add(tag='loss', simple_value=value_lv)
    value_av = None
    av = tf.Summary()
    av.value.add(tag='accuracy', simple_value=value_av)

    for n in range(NUM_STEPS):
        print("step {} out of {}".format(n, NUM_STEPS))
        global_step = sess.run(tf.train.get_global_step())
        if n % NUM_STEPS_PER_EPOCH == 0 or n == NUM_STEPS - 1:
            print("test time")
            # the following code is just to calculate accuracy and loss on the entire validation set
            acc_vld, lss_vld = 0, 0
            for i in range(NUM_TEST_STEPS_PER_EPOCH):
                summary, acc, lss, = sess.run([merged, accuracy, loss], feed_dict={handle: vh})
                acc_vld += acc
                lss_vld += lss
            acc_vld /= NUM_TEST_STEPS_PER_EPOCH
            lss_vld /= NUM_TEST_STEPS_PER_EPOCH

            av.value[0].simple_value = acc_vld
            lv.value[0].simple_value = lss_vld
            vld_writer.add_summary(av, global_step=global_step)
            vld_writer.add_summary(lv, global_step=global_step)

            # summary = sess.run(merged, feed_dict={handle: vh})
            # vld_writer.add_summary(summary, global_step=global_step)
            saver.save(sess, os.path.join(MODEL_FOLDER, "model.ckpt"))
        else:
            summary, _ = sess.run([merged, train_step], feed_dict={handle: th})
            if np.random.random() > 0:
                trn_writer.add_summary(summary, global_step=global_step)
