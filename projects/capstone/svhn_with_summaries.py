from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pdb import set_trace as bp
from svhn_data import load_svhn_data

#Constants
LEARNING_RATE = .001
DROP_OUT = 0.9
MAX_STEPS = 2000
TENSORBOARD_SUMMARIES_DIR = '/tmp/svhn_logs'
IMG_ROWS = 32
IMG_COLS = 32
NUM_CHANNELS = 3
NUM_LABELS = 10
BATCH_SIZE = 1024

'''Load training files from svhn_data.py utility.'''
train_X, train_y = load_svhn_data("training")
valid_X, valid_y = load_svhn_data("validation")
test_X, test_y = load_svhn_data("testing")

# Create a multilayer model.
# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

def model(x, y_):
  #create a 2-layer model with dropout
  hidden1 = nn_layer(x, 3072, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)

  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
  return accuracy, keep_prob, train_step

def predict(accuracy, x, y_, keep_prob, train_step):
  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/train',sess.graph)
    test_writer = tf.train.SummaryWriter(TENSORBOARD_SUMMARIES_DIR + '/test')
    tf.initialize_all_variables().run()
    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries
    for i in range(MAX_STEPS):
      if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(x, y_, keep_prob, i, False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
      else:  # Record train set summaries, and train
        if i % 100 == 99:  # Record execution stats
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(x, y_,  keep_prob, i, True), options=run_options, run_metadata=run_metadata)
          train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
          train_writer.add_summary(summary, i)
          print('Adding run metadata for', i)
        else:  # Record a summary
          summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(x, y_, keep_prob, i, True))
          train_writer.add_summary(summary, i)

def batched_data(idx, imgs, labels, ds):
  """Return the next `batch_size` examples from this data set."""
  start = idx + BATCH_SIZE
  end = start + BATCH_SIZE
  batched_images = imgs[start:end]
  batched_labels = labels[start:end]
  return batched_images, batched_labels

def feed_dict(x, y_, keep_prob, idx, train):
  #TODO fix batching
  """Creates a TensorFlow feed_dict: maps data into Tensor placeholders."""
  if train:
    xs, ys = batched_data(idx, train_X, train_y, "TRAIN")
    k = DROP_OUT
  else:
    xs, ys = batched_data(idx, test_X, test_y, "TEST")
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

def prepare_log_dir():
  '''Clears the log files then creates new directories to place the 
     tensorbard log file.'''
  if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
    tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
  tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)

def main(_):
  prepare_log_dir()
  # Import data

  # Input placeholders
  with tf.name_scope('input'):
    row_len = IMG_ROWS * IMG_COLS * NUM_CHANNELS
    x = tf.placeholder(tf.float32, [BATCH_SIZE, row_len], name='Images')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_LABELS], name='Labels')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, IMG_ROWS, IMG_COLS, NUM_CHANNELS])
    tf.image_summary('input', image_shaped_input, 100)

  accuracy, keep_prob, train_step = model(x, y_)
  predict(accuracy, x, y_, keep_prob, train_step)

if __name__ == '__main__':
  tf.app.run()