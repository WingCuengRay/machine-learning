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
#from svhn_model import classification_head
from pdb import set_trace as bp

TENSORBOARD_SUMMARIES_DIR = '/tmp/svhn_classifier_logs'
NUM_LABELS = 10
BATCH_SIZE = 256
NUM_EPOCHS = 100
IMG_ROWS = 32
IMG_COLS = 32
NUM_CHANNELS = 3

DEPTH_3 =  256
DEPTH_4 = 512
DROPOUT = 0.75
num_hidden1 = 64
#num_hidden1 = 1024
PATCH_SIZE = 5
DEPTH_1 = 64
DEPTH_2 = 64



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

def fill_feed_dict(data, labels, x, y_, step):
  size = labels.shape[0]
  # Compute the offset of the current minibatch in the data.
  # Note that we could use better randomization across epochs.
  offset = (step * BATCH_SIZE) % (size - BATCH_SIZE)
  batch_data = data[offset:(offset + BATCH_SIZE), ...]
  batch_labels = labels[offset:(offset + BATCH_SIZE)]
  return {x: batch_data, y_: batch_labels}


def _activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def train_classification(train_data, train_labels, valid_data, valid_labels, test_data, test_labels, train_size, saved_weights_path):
  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.Variable(0, trainable=False)    
    
    # This is where training samples and labels are fed to the graph.
    with tf.name_scope('input'):
      X_train = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_ROWS, IMG_COLS, NUM_CHANNELS])
      X_valid = tf.constant(valid_data)
      X_test = tf.constant(test_data)

    with tf.name_scope('image'):
      tf.image_summary('train_input', X_train, 10)
      tf.image_summary('valid_input', X_valid, 10)
      tf.image_summary('test_input', X_test, 10)

    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS])

###########################################
    #Variables
    conv1_weights = tf.get_variable("Weights_1", shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_1],\
             initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv1_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_1]), name='Biases_1')

    conv2_weights = tf.get_variable("Weights_2", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_1, DEPTH_2],\
             initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_2]), name='Biases_2')

    conv3_weights = tf.get_variable("Weights_3", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_2, DEPTH_3],\
             initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv3_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_3]), name='Biases_3')

    cl_l3_weights = tf.get_variable("Classifer_Weights_1", shape=[4096, 384], initializer=tf.contrib.layers.xavier_initializer())
    cl_l3_biases = tf.Variable(tf.constant(0.0, shape=[384]), name='Classifer_Biases_1')

    cl_l4_weights = tf.get_variable("Classifer_Weights_2", shape=[384, 192], initializer=tf.contrib.layers.xavier_initializer())
    cl_l4_biases = tf.Variable(tf.constant(0.0, shape=[192]), name='Classifer_Biases_2')

    cl_out_weights = tf.get_variable("Classifer_Weights_3", shape=[192, 10], initializer=tf.contrib.layers.xavier_initializer())
    cl_out_biases = tf.Variable(tf.constant(0.0, shape=[10]), name='Classifer_Biases_3')

    def convolution_model(data_node):
      '''Layer 1'''
      with tf.variable_scope('conv_1') as scope:
        conv1 = tf.nn.conv2d(data_node, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        bias1 = tf.nn.bias_add(conv1, conv1_biases)
        relu1 = tf.nn.relu(bias1,  name=scope.name)
        _activation_summary(relu1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1,2,2,1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        #print("Pool 1 shape", pool1.get_shape().as_list())

      '''Layer 2'''
      with tf.variable_scope('conv_2') as scope:
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        bias2 = tf.nn.bias_add(conv2, conv2_biases)
        relu2 = tf.nn.relu(bias2, name=scope.name)
        _activation_summary(relu2)
        norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')
      return pool2


    def classification_head(X, train=False):
      conv_layer = convolution_model(X)
      shape = conv_layer.get_shape().as_list()
      dim = shape[1] * shape[2] * shape[3]

      #apply dopout to training.
      if train == True:
        print("using drop out")
        fc_out = tf.nn.dropout(conv_layer, DROPOUT)
      else:
         print("not using dropout")

      #Fully Connected Layer
      with tf.variable_scope('fully_connected_1') as scope:
        fc1 = tf.reshape(conv_layer, [shape[0], -1])
        fc1 = tf.add(tf.matmul(fc1, cl_l3_weights), cl_l3_biases)
        fc_out = tf.nn.relu(fc1, name=scope.name)
        _activation_summary(fc_out)
     
      #Fully Connected Layer
      with tf.variable_scope('fully_connected_2') as scope:
        fc2 = tf.add(tf.matmul(fc_out, cl_l4_weights), cl_l4_biases)
        fc2_out = tf.nn.relu(fc2, name=scope.name)
        _activation_summary(fc2_out)

      with tf.variable_scope("softmax_linear") as scope:
        logits = tf.matmul(fc2_out, cl_out_weights) + cl_out_biases
        _activation_summary(logits)

      '''output class scores'''
      return logits #, weights
##############################################################################

    # Training computation: logits + cross-entropy loss.
    logits = classification_head(X_train, True)  
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))

    learning_rate = tf.train.exponential_decay(0.05, global_step*BATCH_SIZE, train_size, 0.975, staircase=True)
    tf.scalar_summary('learning_rate', learning_rate)
    '''Optimizer: set up a variable that's incremented 
      once per batch and controls the learning rate decay.'''
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(classification_head(X_train))
    valid_prediction = tf.nn.softmax(classification_head(X_valid))
    test_prediction = tf.nn.softmax(classification_head(X_test))

    init_op = tf.initialize_all_variables()
    
    #Accuracy ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False)) as sess:
      
      # Restore variables from disk.
      if(saved_weights_path):
        saver.restore(sess, saved_weights_path)
        print("Model restored.")

      sess.run(init_op)
      # Run all the initializers to prepare the trainable parameters.
      
      #Add histograms for trainable variables.
      for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

      #Add accuracy to tesnosrboard
      with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction,  tf.int64))
        tf.scalar_summary('accuracy', accuracy)

      #Prepare vairables for the tensorboard
      merged = tf.merge_all_summaries()

      train_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/train', sess.graph)
      #valid_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/validation')
      #test_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/test')

      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()

      # Loop through training steps.
      for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
        # Run the graph and fetch some of the nodes.
        # This dictionary maps the batch data (as a numpy array) to the
        feed_dict = fill_feed_dict(train_data, train_labels, X_train, y_, step)
        _, l, lr, preds = sess.run([optimizer, loss, learning_rate, train_prediction], feed_dict = feed_dict)
        duration = time.time() - start_time

        if step %  1000 == 0:
          # valid_feed_dict = fill_feed_dict(valid_data, valid_labels, X_valid, y_, step)
          # valid_summary, _, l, lr, valid_predictions = sess.run([merged, optimizer, loss, learning_rate, valid_prediction], feed_dict=valid_feed_dict, options=run_options, run_metadata=run_metadata)
          # valid_writer.add_run_metadata(run_metadata, 'step%03d' % step)
          # valid_writer.add_summary(valid_summary, step)

          train_summary, _, l, lr, predictions = sess.run([merged, optimizer, loss, learning_rate, train_prediction], feed_dict = feed_dict)
          train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
          train_writer.add_summary(train_summary, step)

          print('Adding run metadata for', step)
          print('Validation Accuracy: %.2f%%' % error_rate(valid_prediction.eval(), valid_labels))

        if step % 100 == 0:
           elapsed_time = time.time() - start_time
           start_time = time.time()
           examples_per_sec = BATCH_SIZE / duration
           format_str = ('%s: step %d, loss = %.2f  learning rate = %.6f  (%.1f examples/sec; %.2f ''sec/batch)')
           print (format_str % (datetime.now(), step, l, lr, examples_per_sec, duration))
           train_error_rate = error_rate(preds, feed_dict[y_])
           print('Mini-Batch Accuracy: %.2f%%' % train_error_rate)
           sys.stdout.flush()

      #Save the variables to disk.
      save_path = saver.save(sess, "classifier.ckpt")
      print("Model saved in file: %s" % save_path)
      # test_feed_dict = fill_feed_dict(test_data, test_labels, X, y_, step)
      # _, l, lr, test_predictions = sess.run([optimizer, loss, learning_rate, prediction], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
      # test_summary, _, l, lr, test_predictions = sess.run([merged, optimizer, loss, learning_rate, prediction], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
      # test_writer.add_run_metadata(run_metadata, 'step%03d' % step)
      # test_writer.add_summary(test_summary, step)
      #test_error_rate = error_rate(test_predictions, test_feed_dict[y_])
      #print('Test Accuracy: %.2f%%' % test_error_rate)
      print('Test Accuracy: %.2f%%' % error_rate(test_prediction.eval(), test_labels))
      
      train_writer.close()
      #valid_writer.close()
      #test_writer.close()


def main(saved_weights_path):
  prepare_log_dir()
  train_data, train_labels= load_svhn_data("train", "cropped")
  valid_data, valid_labels = load_svhn_data("valid", "cropped")
  test_data, test_labels = load_svhn_data("test", "cropped")
  #temp
  test_data = test_data[0:1000]
  test_labels = valid_labels[0:1000]
  
  train_size = train_labels.shape[0]
  saved_weights_path = None
  train_classification(train_data, train_labels, valid_data, valid_labels, test_data, test_labels, train_size, saved_weights_path)
  

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