from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import sys
import os
import tensorflow as tf

from svhn_data import load_svhn_data
from datetime import datetime
#from svhn_model import regression_head
from pdb import set_trace as bp

#Run Options 
BATCH_SIZE = 32
NUM_EPOCHS = 100
TENSORBOARD_SUMMARIES_DIR = '/tmp/svhn_regression_logs'

#Image Settings 
NUM_CHANNELS = 3
IMG_HEIGHT = 64
IMG_WIDTH = 64

#Label Settings
NUM_LABELS = 11
LABELS_LEN = 6

#Hyper Parameters
PATCH_SIZE = 5
DEPTH_1 = 16
DEPTH_2 = 32
DEPTH_3 =  64
num_hidden1 = 128

LEARN_RATE = 0.05
DECAY_STEP = 10000
DECAY_RATE = 0.95


def prepare_log_dir():
  '''Clears the log files then creates new directories to place the 
     tensorbard log file.'''
  if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
    tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
  tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)


def fill_feed_dict(data, labels, x, y_, step):
  set_size = labels.shape[0]
  # Compute the offset of the current minibatch in the data.
  # Note that we could use better randomization across epochs.
  offset = (step * BATCH_SIZE) % (set_size - BATCH_SIZE)
  batch_data = data[offset:(offset + BATCH_SIZE), ...]
  batch_labels = labels[offset:(offset + BATCH_SIZE)]
  return {x: batch_data, y_: batch_labels}


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])


def _activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def train_regressor(train_data, train_labels, valid_data, valid_labels, test_data, test_labels, train_size, saved_weights_path):
  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    # This is where training samples and labels are fed to the graph.
    with tf.name_scope('input'):
      X_train = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
      X_valid = tf.constant(valid_data)
      #X_test = tf.constant(test_data)
    
    with tf.name_scope('image'):
      tf.image_summary('input', X_train, 10)

    y_ = tf.placeholder(tf.int32, shape=(BATCH_SIZE, LABELS_LEN))


    ##############################
    conv1_weights = tf.get_variable("W1", shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_1])
    conv1_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_1]), name='B1')

    conv2_weights = tf.get_variable("W2", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_1, DEPTH_2])
    conv2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_2]), name='B2')

    conv3_weights = tf.get_variable("W3", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_2, DEPTH_3])
    conv3_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_3]), name='B3')

    conv4_weights = tf.get_variable("W4", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_3, num_hidden1])
    conv4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name='B4')

    reg1_weights = tf.get_variable("WS1", shape=[num_hidden1, NUM_LABELS])
    reg1_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS1')

    reg2_weights = tf.get_variable("WS2", shape=[num_hidden1, NUM_LABELS])
    reg2_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS2')

    reg3_weights = tf.get_variable("WS3", shape=[num_hidden1, NUM_LABELS])
    reg3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS3')

    reg4_weights = tf.get_variable("WS4", shape=[num_hidden1, NUM_LABELS])
    reg4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS4')

    reg5_weights = tf.get_variable("WS5", shape=[num_hidden1, NUM_LABELS])
    reg5_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS5')


    # Model.
    def regression_head(data, graph, keep_prob=1.0):
      with graph.as_default():
        #######################

        with tf.variable_scope('conv_1') as scope:
          conv = tf.nn.conv2d(data, conv1_weights, [1,1,1,1], 'VALID', name='C1')
          hidden = tf.nn.relu(conv + conv1_biases)
          lrn = tf.nn.local_response_normalization(hidden)
          sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S2')
          _activation_summary(sub)
        
        with tf.variable_scope('conv_2') as scope:
          conv = tf.nn.conv2d(sub, conv2_weights, [1,1,1,1], padding='VALID', name='C3')
          hidden = tf.nn.relu(conv + conv2_biases)
          lrn = tf.nn.local_response_normalization(hidden)
          sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S4')
          _activation_summary(sub)

        with tf.variable_scope('conv_3') as scope:
          conv = tf.nn.conv2d(sub, conv3_weights, [1,1,1,1], padding='VALID', name='C5')
          hidden = tf.nn.relu(conv + conv3_biases)
          lrn = tf.nn.local_response_normalization(hidden)
          sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S5')
          _activation_summary(sub)

        with tf.variable_scope('full_connected_1') as scope:
          conv = tf.nn.conv2d(sub, conv4_weights, [1,1,1,1], padding='VALID', name='C5')
          hidden = tf.nn.relu(conv + conv4_biases)
          hidden = tf.nn.dropout(hidden, keep_prob)
          _activation_summary(hidden)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

        with tf.variable_scope('Output') as scope:
          logits_1 = tf.matmul(reshape, reg1_weights) + reg1_biases
          logits_2 = tf.matmul(reshape, reg2_weights) + reg2_biases
          logits_3 = tf.matmul(reshape, reg3_weights) + reg3_biases
          logits_4 = tf.matmul(reshape, reg4_weights) + reg4_biases
          logits_5 = tf.matmul(reshape, reg5_weights) + reg5_biases
        
        return [logits_1, logits_2, logits_3, logits_4, logits_5]
    ############################

    [logits_1, logits_2, logits_3, logits_4, logits_5] = regression_head(X_train, graph, True)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_1, y_[:,1])) +\
      tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_2, y_[:,2])) +\
      tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_3, y_[:,3])) +\
      tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_4, y_[:,4])) +\
      tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_5, y_[:,5]))

    learning_rate = tf.train.exponential_decay(LEARN_RATE, global_step, DECAY_STEP, DECAY_RATE)
    tf.scalar_summary('learning_rate', learning_rate)

    # Optimizer: set up a variable that's incremented once per batch
    with tf.name_scope('train'):
      optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_prediction = tf.pack([tf.nn.softmax(regression_head(X_train, graph)[0]),\
                        tf.nn.softmax(regression_head(X_train, graph)[1]),\
                        tf.nn.softmax(regression_head(X_train, graph)[2]),\
                        tf.nn.softmax(regression_head(X_train, graph)[3]),\
                        tf.nn.softmax(regression_head(X_train, graph)[4])])

    valid_prediction= tf.pack([tf.nn.softmax(regression_head(X_valid, graph)[0]),\
                        tf.nn.softmax(regression_head(X_valid, graph)[1]),\
                        tf.nn.softmax(regression_head(X_valid, graph)[2]),\
                        tf.nn.softmax(regression_head(X_valid, graph)[3]),\
                        tf.nn.softmax(regression_head(X_valid, graph)[4])])
  
    # test_prediction = tf.pack([tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[0]),\
    #                  tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[1]),\
    #                  tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[2]),\
    #                  tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[3]),\
    #                  tf.nn.softmax(model(tf_test_dataset, 1.0, shape)[4])])
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    start_time = time.time()
    # Create a local session to run the training.
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
   
      init_op = tf.initialize_all_variables()
      # Restore variables from disk.
      if(saved_weights_path):
        saver.restore(sess, saved_weights_path)
        print("Model restored.")
      
      reader = tf.train.NewCheckpointReader("classifier.ckpt")
      reader.get_variable_to_shape_map()

      # Run all the initializers to prepare the trainable parameters.
      sess.run(init_op)

      #Add histograms for trainable variables.
      for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

      #Add accuracy to tesnosrboard
      # with tf.name_scope('accuracy'):
      #   with tf.name_scope('correct_prediction'):
      #     bp()
      #     #correct_prediction = (np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])
      #     #return (100.0 * 
      #   with tf.name_scope('accuracy'):
      #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      #   tf.scalar_summary('accuracy', accuracy)

      #Prepare vairables for the tensorboard
      merged = tf.merge_all_summaries()
      train_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/train', sess.graph)
      #valid_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/validation')
      #test_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/test')
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()

      # Loop through training steps.
      for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
        duration = time.time() - start_time
        examples_per_sec = BATCH_SIZE / duration 
        
        # Run the graph and fetch some of the nodes.
        # This dictionary maps the batch data (as a numpy array) to the
        
        train_feed_dict = fill_feed_dict(train_data, train_labels, X_train, y_, step)
        #train_summary, _, l, predictions = sess.run([merged, optimizer, loss, train_prediction], feed_dict = train_feed_dict)
        _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction], feed_dict = train_feed_dict)

        train_batched_labels = train_feed_dict.values()[1]
  
        # if step % 100 == 0:
        #   valid_feed_dict = fill_feed_dict(valid_data, valid_labels, X_valid, y_, step)
        #   valid_batch_labels = valid_feed_dict.values()[1]
          
        #   valid_summary, _, l, lr, valid_predictions = sess.run([merged, optimizer, loss, learning_rate, valid_prediction], 
        #     feed_dict=valid_feed_dict, options=run_options, run_metadata=run_metadata)
          
        #   #valid_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        #   #valid_writer.add_summary(valid_summary, step)

        #   #   train_summary, _, l, lr, predictions = sess.run([merged, optimizer, loss, learning_rate, prediction], 
        #   #     feed_dict = train_feed_dict)
        #   #train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        #   #train_writer.add_summary(train_summary, step)

        #   print('Adding run metadata for', step)
        #   print('Validation Accuracy: %.2f%%' % accuracy(valid_predictions, valid_batch_labels[:,1:6]))
        
        if step % 100 == 0:
          elapsed_time = time.time() - start_time
          start_time = time.time()
          
          format_str = ('%s: step %d, loss = %.2f  learning rate = %.2f  (%.1f examples/sec; %.3f ''sec/batch)')
          print (format_str % (datetime.now(), step, l, lr, examples_per_sec, duration))
          
          print('Minibatch accuracy: %.2f%%' % accuracy(predictions, train_batched_labels[:,1:6]))
          sys.stdout.flush()

      # Save the variables to disk.

      # test_feed_dict = fill_feed_dict(test_data, test_labels, x, y_, step)
      # test_batch_labels = test_feed_dict.values()[1]
      # _, l, lr, test_predictions = sess.run([optimizer, loss, learning_rate, prediction], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
      # test_summary, _, l, lr, test_predictions = sess.run([merged, optimizer, loss, learning_rate, prediction], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
      # test_writer.add_run_metadata(run_metadata, 'step%03d' % step)
      # test_writer.add_summary(test_summary, step)

      save_path = saver.save(sess, "/tmp/regression.ckpt")
      print("Model saved in file: %s" % save_path)

      train_writer.close()
      #valid_writer.close()
      #test_writer.close()
      #print('Test accuracy: %.2f%%' % accuracy(test_predictions, test_batch_labels[:,1:6]))



def main(saved_weights_path):
  prepare_log_dir()
  train_data, train_labels= load_svhn_data("train", "full")
  valid_data, valid_labels= load_svhn_data("valid", "full")
  test_data, test_labels= load_svhn_data("test", "full")
  print("TrainData", train_data.shape)
  print("Valid Data", valid_data.shape)
  print("Test Data", test_data.shape)
  train_size = len(train_labels)
  train_regressor(train_data, train_labels, valid_data, valid_labels, test_data, test_labels, train_size, saved_weights_path)


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