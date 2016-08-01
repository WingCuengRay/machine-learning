from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import numpy
import sys
import os
import tensorflow as tf

from svhn_data import load_svhn_data
from svhn_model import model
from pdb import set_trace as bp
#from mnist_data import load_mnist_data

TENSORBOARD_SUMMARIES_DIR = '/tmp/svhn_convo_logs'
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 2
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
IMG_ROWS = 32
IMG_COLS = 32
NUM_CHANNELS = 3

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)
  return accuracy.eval() * 100

def prepare_log_dir():
  '''Clears the log files then creates new directories to place the 
     tensorbard log file.'''
  if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
    tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
  tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)

# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def eval_in_batches(data, sess, eval_prediction, eval_data):
  """Get all predictions for a dataset by running it in small batches."""
  size = data.shape[0]
  if size < BATCH_SIZE:
    raise ValueError("batch size for evals larger than dataset: %d" % size)
  predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
  
  for begin in xrange(0, size, BATCH_SIZE):
    end = begin + BATCH_SIZE
    if end <= size:
      predictions[begin:end, :] = sess.run(
          eval_prediction,
          feed_dict={eval_data: data[begin:end, ...]})
    else:
      batch_predictions = sess.run(
          eval_prediction,
          feed_dict={eval_data: data[-BATCH_SIZE:, ...]})
      predictions[begin:, :] = batch_predictions[begin - size:, :]
  return predictions

def train(initial_weights=None):
  pass

def main(saved_weights_path):
  print("SWP", saved_weights_path)
  prepare_log_dir()
  
  train_data, train_labels= load_svhn_data("training")
  valid_data, valid_labels = load_svhn_data("validation")
  test_data, test_labels = load_svhn_data("testing")
  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  with tf.name_scope('input'):
    tf_data_node = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_ROWS, IMG_COLS, NUM_CHANNELS])
    tf_labels_node = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS])
    #eval_data = tf.placeholder(tf.float32,  shape=(EVAL_BATCH_SIZE, IMG_ROWS, IMG_COLS, NUM_CHANNELS))
    #eval_data = tf_data_node

  with tf.name_scope('image'):
    tf.image_summary('input', tf_data_node, 100)

  # Training computation: logits + cross-entropy loss.
  logits, params = model(tf_data_node, True)
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  sm_pred, params = model(tf_data_node)
  eval_prediction = tf.nn.softmax(sm_pred)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels_node))
  # L2 regularization for the fully connected parameters.
  l3_weights = params[4]
  l3_biases = params[5]
  l4_weights = params[6]
  l4_biases = params[7]
  regularizers = (tf.nn.l2_loss(l3_weights) + tf.nn.l2_loss(l3_biases) + tf.nn.l2_loss(l4_weights) + tf.nn.l2_loss(l4_biases))
  
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
  # Use simple momentum for the optimization.
  tf.scalar_summary('learning_rate', learning_rate)
  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
  # Predictions for the current training minibatch.

  

  #tf.initialize_all_variables().run()
  init_op = tf.initialize_all_variables()

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  
  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    
    # Restore variables from disk.
    if(saved_weights_path):
      saver.restore(sess, saved_weights_path)
      print("Model restored.")

    sess.run(init_op)
    #Run all the initializers to prepare the trainable parameters.
    #Prepare vairables for the tensorboard
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/train',sess.graph)
    test_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/test')

    print('Initialized!')
    # Loop through training steps.

    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {tf_data_node: batch_data, tf_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      
      summary, _, l, lr, predictions = sess.run([merged, optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
      
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
        print("Of Step:", int(NUM_EPOCHS * train_size))
       
        print('Minibatch loss: %.2f, learning rate: %.6f' % (l, lr))
        print('Minibatch Accuracy: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation Accuracy: %.2f%%' % error_rate(eval_in_batches(valid_data, sess, eval_prediction, tf_data_node), valid_labels))
        sys.stdout.flush()
    
      if step % 10 == 0:  # Record summaries and test-set accuracy
        test_writer.add_summary(summary, step)

      if step % EVAL_FREQUENCY == 99:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _, l, lr, predictions = sess.run([merged, optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        train_writer.add_summary(summary, step)
        print('Adding run metadata for', step)
        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)

    # Finally print the result!
    train_writer.close()
    test_writer.close()
    test_error = error_rate(eval_in_batches(test_data, sess, eval_prediction, tf_data_node), test_labels)
    #print('Test Accuracy: %.2f%%' % test_error)

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