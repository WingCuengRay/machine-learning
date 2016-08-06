from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy
import sys
import os
import tensorflow as tf

from svhn_data import load_svhn_data
from datetime import datetime
from svhn_model import classification_head
from pdb import set_trace as bp

TENSORBOARD_SUMMARIES_DIR = '/tmp/svhn_convo_logs'
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 1024
#NUM_EPOCHS = 256
NUM_EPOCHS = 50

def error_rate(predictions, labels):
  correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy.eval() * 100

def prepare_log_dir():
  '''Clears the log files then creates new directories to place the 
     tensorbard log file.'''
  if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
    tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
  tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)

def fill_feed_dict(train_data, train_labels, x, y_, step):
  train_size = train_labels.shape[0]
  # Compute the offset of the current minibatch in the data.
  # Note that we could use better randomization across epochs.
  offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)

  batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
  batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

  return {x: batch_data, y_: batch_labels}

def train(train_data, train_labels, valid_data, valid_labels, test_data, test_labels, train_size, saved_weights_path):
  # This is where training samples and labels are fed to the graph.
  y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS])

  # Training computation: logits + cross-entropy loss.
  x, logits, params = classification_head(BATCH_SIZE, True)
  prediction = tf.nn.softmax(logits)

  # L2 regularization for the fully connected parameters.
  l3_weights = params[4]
  l3_biases = params[5]
  l4_weights = params[6]
  l4_biases = params[7]
  regularizers = (tf.nn.l2_loss(l3_weights) + tf.nn.l2_loss(l3_biases) + tf.nn.l2_loss(l4_weights) + tf.nn.l2_loss(l4_biases))

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))

  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)

  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  tf.scalar_summary('learning_rate', learning_rate)

  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

  #tf.initialize_all_variables().run()
  init_op = tf.initialize_all_variables()
  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    # Restore variables from disk.
    if(saved_weights_path):
      saver.restore(sess, saved_weights_path)
      print("Model restored.")

    sess.run(init_op)
    # Run all the initializers to prepare the trainable parameters.
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
     tf.histogram_summary(var.op.name, var)

    #Add accuracy to tesnosrboard
    with tf.name_scope('accuracy'):
     with tf.name_scope('correct_prediction'):
       correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
     with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     tf.scalar_summary('accuracy', accuracy)

    #Prepare vairables for the tensorboard
    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/train', sess.graph)
    valid_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/validation')
    test_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/test')

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # Loop through training steps.
    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      # Run the graph and fetch some of the nodes.
      # This dictionary maps the batch data (as a numpy array) to the
      feed_dict = fill_feed_dict(train_data, train_labels, x, y_, step)
      _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, prediction], feed_dict = feed_dict)
      duration = time.time() - start_time

      if step %  1000 == 0:
        valid_feed_dict = fill_feed_dict(valid_data, valid_labels, x, y_, step)
        
        valid_summary, _, l, lr, valid_predictions = sess.run([merged, optimizer, loss, learning_rate, prediction], feed_dict=valid_feed_dict, options=run_options, run_metadata=run_metadata)
        valid_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        valid_writer.add_summary(valid_summary, step)
        
        train_summary, _, l, lr, predictions = sess.run([merged, optimizer, loss, learning_rate, prediction], feed_dict = feed_dict)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        train_writer.add_summary(train_summary, step)

        print('Adding run metadata for', step)
        print('Validation Accuracy: %.2f%%' % error_rate(valid_predictions, feed_dict[y_]))

      if step % 10 == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        examples_per_sec = BATCH_SIZE / duration
        format_str = ('%s: step %d, loss = %.2f  learning rate = %.2f  (%.1f examples/sec; %.3f ''sec/batch)')
        print (format_str % (datetime.now(), step, l, lr, examples_per_sec, duration))
        sys.stdout.flush()

    #Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    test_feed_dict = fill_feed_dict(test_data, test_labels, x, y_, step)
    _, l, lr, test_predictions = sess.run([optimizer, loss, learning_rate, prediction], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
    test_summary, _, l, lr, test_predictions = sess.run([merged, optimizer, loss, learning_rate, prediction], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
    test_writer.add_run_metadata(run_metadata, 'step%03d' % step)
    test_writer.add_summary(test_summary, step)

    train_writer.close()
    valid_writer.close()
    test_writer.close()
    #test_error = error_rate(eval_in_batches(test_data, sess, eval_prediction, tf_data_node), test_labels)
    #print('Test Accuracy: %.2f%%' % test_error)

def main(saved_weights_path):
  prepare_log_dir()
  #train_data, train_labels= load_svhn_data("training", "cropped")
  #valid_data, valid_labels = load_svhn_data("validation", "cropped")
  #test_data, test_labels = load_svhn_data("testing", "cropped")
  
  #load full sized data
  train_data_full, train_labels_full= load_svhn_data("training", "full")
  #valid_data_full, valid_labels_full = load_svhn_data("validation", "full")
  #test_data_full, test_labels_full = load_svhn_data("testing", "full")

  #train_size = train_labels.shape[0]
  #train(train_data, train_labels, valid_data, valid_labels, test_data, test_labels, train_size, saved_weights_path)
  

if __name__ == '__main__':
  saved_weights_path = None
  if len(sys.argv) > 1:
    print("Loading Saved Checkpoints From:", sys.argv[1])
    if os.path.isfile(sys.argv[1]):
      saved_weights_path = sys.argv[1]
    else:
      raise EnvironmentError("I'm sorry, Dave. I'm afraid I can't do that...")
  else:
    print("Starting without Saved Weights.")
  main(saved_weights_path)