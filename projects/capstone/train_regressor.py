# -*- coding:utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import sys
import os
import tensorflow as tf
import random

from svhn_data import load_svhn_data
from svhn_model import regression_head

from datetime import datetime

# Run Options
BATCH_SIZE = 64
NUM_EPOCHS = 128        # EPOCH　代表整个数据集被重复训练的多少次
#TENSORBOARD_SUMMARIES_DIR = '/home/ray/svhn_regression_logs'
TENSORBOARD_SUMMARIES_DIR = '/tmp/svhn_regression_logs'

# Image Settings
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CHANNELS = 3

# Label Settings
NUM_LABELS = 11
LABELS_LEN = 6

# LEARING RATE HYPER PARAMS
LEARN_RATE = 0.075
DECAY_RATE = 0.95
STAIRCASE = True


def prepare_log_dir():
    '''Clears the log files then creates new directories to place
        the tensorbard log file.'''
    if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
    tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)

'''
def fill_feed_dict(data, labels, x, y_, step):
    set_size = labels.shape[0]
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.

    # 根据 step 的递增，有规律（有序）地取不同的 BATCH
    offset = (step * BATCH_SIZE) % (set_size - BATCH_SIZE)
    batch_data = data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = labels[offset:(offset + BATCH_SIZE)]
    return {x: batch_data, y_: batch_labels}
'''
def fill_feed_dict(data, labels, x, y_, step):
    set_size = labels.shape[0]
    offset = random.randint(0, set_size-BATCH_SIZE)

    batch_data = data[offset:(offset+BATCH_SIZE), ...]
    batch_labels = labels[offset:(offset+BATCH_SIZE)]

    return {x:batch_data, y_:batch_labels}


def train_regressor(train_data, train_labels, valid_data, valid_labels,
                    test_data, test_labels, train_size, saved_weights_path):
    global_step = tf.Variable(0, trainable=False)
    # This is where training samples and labels are fed to the graph.
    with tf.name_scope('input'):
        images_placeholder = tf.placeholder(tf.float32,
                                            shape=(BATCH_SIZE, IMG_HEIGHT,
                                                   IMG_WIDTH, NUM_CHANNELS))

    with tf.name_scope('image'):
        tf.summary.image('input', images_placeholder, 10)

    labels_placeholder = tf.placeholder(tf.int32,
                                        shape=(BATCH_SIZE, LABELS_LEN))

    [logits_1, logits_2, logits_3, logits_4, logits_5] = regression_head(images_placeholder, True)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=labels_placeholder[:, 1])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=labels_placeholder[:, 2])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=labels_placeholder[:, 3])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=labels_placeholder[:, 4])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=labels_placeholder[:, 5]))

    learning_rate = tf.train.exponential_decay(LEARN_RATE, global_step*BATCH_SIZE, train_size, DECAY_RATE)
    tf.summary.scalar('learning_rate', learning_rate)

    # Optimizer: set up a variable that's incremented once per batch
    with tf.name_scope('train'):
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # prediciton: [5 * N * NUM_LABELS]
    prediction = tf.stack([tf.nn.softmax(regression_head(images_placeholder)[0]),
                                tf.nn.softmax(regression_head(images_placeholder)[1]),
                                tf.nn.softmax(regression_head(images_placeholder)[2]),
                                tf.nn.softmax(regression_head(images_placeholder)[3]),
                                tf.nn.softmax(regression_head(images_placeholder)[4])])

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    start_time = time.time()
    # Create a local session to run the training.
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init_op = tf.initialize_all_variables()
        # Run all the initializers to prepare the trainable parameters.
        sess.run(init_op)

        
        # Restore variables from disk.
        if(saved_weights_path):
            saver.restore(sess, saved_weights_path)
        print("Model restored.")

        #reader = tf.train.NewCheckpointReader("classifier.ckpt")
        #reader.get_variable_to_shape_map()

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                # transpose 的 perm 参数: [1,2,0] 代表将原来的第 1 维 transpose 到第　0　维
                # 第 2　维 transpose　到第 1 维， ...

                # best: [N * 11 * 5]
                best = tf.transpose(prediction, [1, 2, 0])  # permute n_steps and batch_size
                lb = tf.cast(labels_placeholder[:, 1:6], tf.int64)

                # tf.argmax(best, 1) --> [N * 5] 每个位置最大概率的数字
                # correct_prediction --> bool
                correct_prediction = tf.equal(tf.argmax(best, 1), lb)
            with tf.name_scope('accuracy'):
                # accuracy 计算的是每个数字的正确率
                # 若要计算每个数字序列的正确率，则去掉后面的 .get_shape().as_list()[0]
                # 相当于 sum(correct_prediction) / (prediction.shape[1]*prediction.shape[0])
                accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / prediction.get_shape().as_list()[1] / prediction.get_shape().as_list()[0]
            tf.summary.scalar('accuracy', accuracy)

        # Prepare vairables for the tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(TENSORBOARD_SUMMARIES_DIR + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(TENSORBOARD_SUMMARIES_DIR + '/validation')
        test_writer = tf.summary.FileWriter(TENSORBOARD_SUMMARIES_DIR + '/test')

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        try:
            # Loop through training steps.
            print("Start training...")
            for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
                duration = time.time() - start_time
                examples_per_sec = BATCH_SIZE / duration

                # Run the graph and fetch some of the nodes.
                # This dictionary maps the batch data (as a numpy array) to the
                train_feed_dict = fill_feed_dict(train_data, train_labels, images_placeholder, labels_placeholder, step)
                _, l, lr, acc, predictions = sess.run([optimizer, loss, learning_rate,
                                                      accuracy, prediction],
                                                      feed_dict=train_feed_dict)

                train_batched_labels = train_feed_dict.values()[1]

                if step % 1000 == 0:
                    # Validation set
                    valid_feed_dict = fill_feed_dict(valid_data, valid_labels, images_placeholder, labels_placeholder, step)
                    valid_batch_labels = valid_feed_dict.values()[1]

                    valid_summary, _, l, lr, valid_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy],
                    feed_dict=valid_feed_dict, options=run_options, run_metadata=run_metadata)
                    print('Validation Accuracy: %.2f' % valid_acc)
                    valid_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    valid_writer.add_summary(valid_summary, step)

                    # Test set
                    test_feed_dict = fill_feed_dict(test_data, test_labels, images_placeholder, labels_placeholder, step)
                    test_summary, _, l, lr, test_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy],
                    feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
                    print('Test Accuracy: %.2f' % test_acc)
                    test_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    test_writer.add_summary(test_summary, step)

                    # Training set
                    train_summary, _, l, lr, train_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy],
                        feed_dict=train_feed_dict)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(train_summary, step)
                    print('Training Set Accuracy: %.2f' % train_acc)
                    print('Adding run metadata for', step)


                elif step % 100 == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    format_str = ('%s: step %d, loss = %.2f  learning rate = %.2f  (%.1f examples/sec; %.3f ''sec/batch)')
                    print (format_str % (datetime.now(), step, l, lr, examples_per_sec, duration))

                    print('Minibatch accuracy2: %.2f' % acc)
                    sys.stdout.flush()

            test_feed_dict = fill_feed_dict(test_data, test_labels, images_placeholder, labels_placeholder, step)
            _, l, lr, test_acc = sess.run([optimizer, loss, learning_rate, accuracy], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
            print('Test accuracy: %.2f' % test_acc)

            # Save the variables to disk.
            save_path = saver.save(sess, "regression.ckpt")
            print("Model saved in file: %s" % save_path)

            train_writer.close()
            valid_writer.close()
            test_writer.close()

        except KeyboardInterrupt:
            save_path = saver.save(sess, "regression.ckpt")
            print("Model saved in file: %s" % save_path)
            train_writer.close()
            valid_writer.close()
            test_writer.close()



def main(saved_weights_path):
    prepare_log_dir()
    train_data, train_labels = load_svhn_data("train", "full")
    valid_data, valid_labels = load_svhn_data("valid", "full")
    test_data, test_labels = load_svhn_data("test", "full")

    print("TrainData", train_data.shape)
    print("Valid Data", valid_data.shape)
    print("Test Data", test_data.shape)

    train_size = len(train_labels)
    train_regressor(train_data, train_labels, valid_data, valid_labels,
                    test_data, test_labels, train_size, saved_weights_path)


if __name__ == '__main__':
    saved_weights_path = None
    if len(sys.argv) > 1:
        print("Loading Saved Checkpoints From:", sys.argv[1])
        if os.path.isfile(sys.argv[1]):
            saved_weights_path = sys.argv[1]
        else:
            raise EnvironmentError("I'm afraid I can't load that file.")
    else:
        print("Starting without Saved Weights.")
    main(saved_weights_path)
