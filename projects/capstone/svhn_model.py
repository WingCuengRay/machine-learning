import tensorflow as tf
from pdb import set_trace as bp

IMG_ROWS = 32
IMG_COLS = 32
NUM_CHANNELS = 3

def _activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def convolution_model(batch_size, train):
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  with tf.name_scope('input'):
    data_node = tf.placeholder(tf.float32, shape=[batch_size, IMG_ROWS, IMG_COLS, NUM_CHANNELS])
    
  with tf.name_scope('image'):
   tf.image_summary('input', data_node, 10)

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 64], stddev=5e-2))
  conv1_biases =  tf.Variable(tf.zeros([64]))

  conv2_weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
  conv2_biases =  tf.Variable(tf.constant(0.1, shape=[64]))
  
  #conv3_weights = tf.Variable
  #conv3_biases = tf.Variable

  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  
  #Layer 1
  if NUM_CHANNELS == 1: #If a greysacle image is sent expand the dimensions
    data_node = tf.expand_dims(data_node, 3)
  
  with tf.variable_scope('conv1') as scope:
    conv1 = tf.nn.conv2d(data_node, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # Bias and rectified linear non-linearity.
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    _activation_summary(relu1)
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  #Layer 2
  with tf.variable_scope('conv2') as scope:
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    _activation_summary(relu2)
    norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  #Layer3
  # with tf.variable_scope('conv3') as scope:
  #   conv3 = tf.nn.conv2d(pool2, conv3_weights, strides = [1,1,1,1,], padding='SAME')
  #   relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
  #   _activation_summary(relu3)
  #   norm3 = tf.nn.lrn(relu3, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
  #   pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  weights = [conv1_weights, conv1_biases, 
            conv2_weights, conv2_biases]

  return data_node, pool2, weights


def classification_head(batch_size, train=False):
  x, conv_layer, conv_vars = convolution_model(batch_size, train=False)

  l3_weights = tf.Variable(tf.truncated_normal(shape=[4096, 384], stddev=0.04))
  l3_biases = tf.Variable(tf.constant(0.1, shape=[384]))

  l4_weights = tf.Variable(tf.truncated_normal(shape=[384, 192], stddev=0.04))
  l4_biases = tf.Variable(tf.constant(0.1, shape=[192]))

  #Layer3
  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(conv_layer, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    l3_weights = tf.Variable(tf.truncated_normal(shape=[dim, 384], stddev=0.04))
    l3_biases = tf.Variable(tf.constant(0.1, shape=[384]))
    l3 = tf.nn.relu(tf.matmul(reshape, l3_weights) + l3_biases)
    _activation_summary(l3)

  #Layer4
  with tf.variable_scope('local4') as scope:
    l4_weights = tf.Variable(tf.truncated_normal(shape=[384, 192], stddev=0.04))
    l4_biases = tf.Variable(tf.constant(0.1, shape=[192]))
    l4 = tf.nn.relu(tf.matmul(l3, l4_weights) + l4_biases)
    _activation_summary(l4)
  
  with tf.variable_scope('softmax_linear') as scope:
    sm_weights = tf.Variable(tf.truncated_normal(shape=[192, 10], stddev=1/192.0))
    sm_biases = tf.Variable(tf.constant(0.1, shape=[10]))
    softmax_linear = tf.add(tf.matmul(l4, sm_weights), sm_biases)
    _activation_summary(softmax_linear)

  weights = conv_vars + [l3_weights, l3_biases, 
            l4_weights, l4_biases]

  #output class scores 
  return x, softmax_linear, weights

def regression_head(batch_size, train=False):
  '''takes any size input and slides a window of size [32x32] 
  across the image in 4x4 strides.'''
  num_runs = 5
  num_labels = 10
  #Densley Connected Layer
  fc1_weights = tf.Variable(tf.truncated_normal(shape=[4096, 384], stddev=0.04))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[384]))

  W_conv1 = tf.reshape(fc1_weights, [8,  32, 128, 2048])
  h_conv1 = tf.nn.relu(conv2d(conv_layer, reshape,
                                stride=(1, 1), padding="VALID") + fc1_biases) 
  #Output Layers
  #5 max outputs, 10 classes
  fc2_weights = weight_variable([2048, 1 + num_runs * num_labels])
  fc2_biases = bias_variable([1 + num_runs * num_labels])

  W_conv2 = tf.reshape(W_fc2, [1, 1, 2048, 1 + num_runs * num_labels])
  h_conv2 = conv2d(h_conv1, W_conv2) + fc2_biases

  x, conv_layer, conv_vars = convolutional_layers(batch_size, train=False)

  #Output Box Coordinates
  pass 