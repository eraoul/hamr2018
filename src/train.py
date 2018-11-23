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
trn_itr = create_tfrecords_iterator(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)
vld_itr = create_tfrecords_iterator(VALIDATION_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

handle = tf.placeholder(tf.string, shape=[])
x, di, y = tf.data.Iterator.from_string_handle(handle, trn_itr.output_types, trn_itr.output_shapes).get_next()
encoder_inputs = tf.unstack(tf.reshape(x, [BATCH_SIZE, TIMESTEPS, VEC_LENGTH]), axis=1)
decoder_inputs = tf.unstack(tf.reshape(di, [BATCH_SIZE, TIMESTEPS, VEC_LENGTH]), axis=1)
targets = tf.reshape(y, [BATCH_SIZE, TIMESTEPS, VEC_LENGTH])

""" Model generation """
logger.info("Creating the model")
logits = seq2seq_train(encoder_inputs, decoder_inputs)

""" Train """
# TODO: Decide a value for the weights of pitch and duration sequence loss
#   Currently they are just ones all the time but two questions arise:
#   1. Should one limit the loss to the first padding token?
#   2. Should one weigh differently different notes, maybe with a larger weight on longer notes or tonic/dominant?
l_pit = logits[:, :, :NUM_MIDI_CLASSES]
t_pit = tf.argmax(targets[:, :, :NUM_MIDI_CLASSES], axis=-1)
w_pit = tf.cast(tf.sequence_mask([TIMESTEPS] * BATCH_SIZE, TIMESTEPS), tf.float32)

l_dur = logits[:, :, NUM_MIDI_CLASSES:]
t_dur = tf.argmax(targets[:, :, NUM_MIDI_CLASSES:], axis=-1)
w_dur = tf.cast(tf.sequence_mask([TIMESTEPS] * BATCH_SIZE, TIMESTEPS), tf.float32)

loss_pit = tf.contrib.seq2seq.sequence_loss(l_pit, t_pit, w_pit)
loss_dur = tf.contrib.seq2seq.sequence_loss(l_dur, t_dur, w_dur)
# TODO: Decide how to combine the two losses.
#   Currently it's just a sum but we might want to do something more elaborate.
loss = loss_pit + loss_dur
tf.summary.scalar('loss_pitch', loss_pit)
tf.summary.scalar('loss_duration', loss_dur)
tf.summary.scalar('loss', loss)

train_step = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.create_global_step())


def _tf_boolean_mean(x):
    tf.assert_type(x, tf.bool)
    return tf.reduce_mean(tf.cast(x, tf.float32))


p_pit = tf.argmax(l_pit, axis=-1)
p_dur = tf.argmax(l_dur, axis=-1)
accuracy_pit = _tf_boolean_mean(tf.equal(p_pit, t_pit))
accuracy_dur = _tf_boolean_mean(tf.equal(p_dur, t_dur))
# TODO: Does is even make sense to have an average accuracy?
accuracy = (accuracy_pit + accuracy_dur) / 2
tf.summary.scalar('accuracy_pitch', accuracy_pit)
tf.summary.scalar('accuracy_duration', accuracy_dur)
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

    # Keep summary of global variables across the validation set
    # TODO: Is there a more elegant and compact way to do this?
    lv, lpv, ldv = tf.Summary(), tf.Summary(), tf.Summary()
    lv.value.add(tag='loss', simple_value=None)
    lpv.value.add(tag='loss_pitch', simple_value=None)
    ldv.value.add(tag='loss_duration', simple_value=None)
    value_av, value_adv, value_apv = None, None, None
    av, apv, adv = tf.Summary(), tf.Summary(), tf.Summary()
    av.value.add(tag='accuracy', simple_value=None)
    apv.value.add(tag='accuracy_pitch', simple_value=None)
    adv.value.add(tag='accuracy_duration', simple_value=None)

    test = sess.run([l_pit, t_pit, w_pit, l_dur, t_dur, w_dur], feed_dict={handle: th})
    print(test)

    for n in range(NUM_STEPS):
        print("step {} out of {}".format(n, NUM_STEPS))
        global_step = sess.run(tf.train.get_global_step())
        if n % NUM_STEPS_PER_EPOCH == 0 or n == NUM_STEPS - 1:
            print("test time")
            # the following code is just to calculate accuracy and loss on the entire validation set
            # TODO: Is there a more elegant and compact way to do this?
            acc_vld, acc_p_vld, acc_d_vld, lss_vld, lss_p_vld, lss_d_vld = 0., 0., 0., 0., 0., 0.
            for i in range(NUM_TEST_STEPS_PER_EPOCH):
                summary, acc, acc_p, acc_d, lss, lss_p, lss_d = sess.run(
                    [merged, accuracy, accuracy_pit, accuracy_dur, loss, loss_pit, loss_dur], feed_dict={handle: vh})
                acc_vld += acc
                acc_p_vld += acc_p
                acc_d_vld += acc_d
                lss_vld += lss
                lss_p_vld += lss_p
                lss_d_vld += lss_d
            acc_vld /= NUM_TEST_STEPS_PER_EPOCH
            acc_p_vld /= NUM_TEST_STEPS_PER_EPOCH
            acc_d_vld /= NUM_TEST_STEPS_PER_EPOCH
            lss_vld /= NUM_TEST_STEPS_PER_EPOCH
            lss_p_vld /= NUM_TEST_STEPS_PER_EPOCH
            lss_d_vld /= NUM_TEST_STEPS_PER_EPOCH

            # Write down the summaries
            av.value[0].simple_value = acc_vld
            apv.value[0].simple_value = acc_p_vld
            adv.value[0].simple_value = acc_d_vld
            lv.value[0].simple_value = lss_vld
            lpv.value[0].simple_value = lss_p_vld
            ldv.value[0].simple_value = lss_d_vld
            vld_writer.add_summary(av, global_step=global_step)
            vld_writer.add_summary(apv, global_step=global_step)
            vld_writer.add_summary(adv, global_step=global_step)
            vld_writer.add_summary(lv, global_step=global_step)
            vld_writer.add_summary(lpv, global_step=global_step)
            vld_writer.add_summary(ldv, global_step=global_step)

            saver.save(sess, os.path.join(MODEL_FOLDER, "model.ckpt"))
        else:
            summary, _ = sess.run([merged, train_step], feed_dict={handle: th})
            if np.random.random() > 0:
                trn_writer.add_summary(summary, global_step=global_step)
