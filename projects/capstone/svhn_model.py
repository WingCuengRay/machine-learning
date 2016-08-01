import tensorflow as tf

LEARNING_RATE = .001
from pdb import set_trace as bp
BATCH_SIZE = 128


# DROP_OUT = 0.9
# MAX_STEPS = 2000
# TENSORBOARD_SUMMARIES_DIR = '/tmp/svhn_logs'
# IMG_ROWS = 32
# IMG_COLS = 32
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_LABELS = 10
BATCH_SIZE = 64
# EVAL_BATCH_SIZE = 128
# SEED = None

#Utilites
def weight_variable(shape):
	"""Create a weight variable with appropriate initialization."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	"""Create a bias variable with appropriate initialization."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, stride=(1, 1), padding='SAME'):
	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding=padding)

def max_pool(x, ksize=(2, 2), stride=(2, 2)):
	return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')

def dropout(layer, keep_prob):
	with tf.name_scope('dropout'):
		tf.scalar_summary('dropout_keep_probability', keep_prob)
		dropped = tf.nn.dropout(layer, keep_prob)
	return dropped

def cross_entropy(y, y_):
	with tf.name_scope('cross_entropy'):
		diff = y_ * tf.log(y)
		with tf.name_scope('total'):
			cross_entropy = -tf.reduce_mean(diff)
		tf.scalar_summary('cross entropy', cross_entropy)
		return cross_entropy


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


# Create a multilayer model.
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
      weights = weight_variable(input_dim)
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('convolution'):     
      preactivate = conv2d(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  #tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, train=False):
  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 64], stddev=5e-2))
  conv1_biases =  tf.Variable(tf.zeros([64]))

  conv2_weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
  conv2_biases =  tf.Variable(tf.constant(0.1, shape=[64]))
  
  l3_weights = tf.Variable(tf.truncated_normal(shape=[4096, 384], stddev=0.04))
  l3_biases = tf.Variable(tf.constant(0.1, shape=[384]))

  l4_weights = tf.Variable(tf.truncated_normal(shape=[384, 192], stddev=0.04))
  l4_biases = tf.Variable(tf.constant(0.1, shape=[192]))

  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  
  #Layer 1
  if NUM_CHANNELS == 1:
    data = tf.expand_dims(data, 3)
  with tf.variable_scope('conv1') as scope:
    conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
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
  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
    dim = reshape.get_shape()[1].value
    #print(dim)
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
    #_activation_summary(softmax_linear)

  return softmax_linear, [conv1_weights, conv1_biases, conv2_weights, conv2_biases, l3_weights, l3_biases, l4_weights, l4_biases]



























# def convolutions(X, y_, keep_prob):
#   '''create a 2-layer model with dropout'''
#   #First Layer
#   #x_expanded = tf.expand_dims(X, 3)
#   conv1 = nn_layer(X, [5, 5, 3, 64], 64, 'layer1', act=tf.nn.relu)
#   pool1 = max_pool(conv1, ksize=(2, 2), stride=(2, 2))
  
#   #Second Layer
#   conv2 = nn_layer(pool1, [5, 5, 64, 64], 64, 'layer2', act=tf.nn.relu)
#   pool2 = max_pool(conv2, ksize=(2, 1), stride=(2, 1))
  

#   #Third Layer
#   #bp()
#   #conv3 = nn_layer(pool2, [5, 5, 64, 64], 64, 'layer3', act=tf.nn.relu)
#   #pool3 = max_pool(conv3, ksize=(2, 2), stride=(2, 2))

#   # cross_ent = cross_entropy(pool2, y_)

#   # with tf.name_scope('train'):
#   #   train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_ent)

#   # acc = accuracy(pool2, y_)  
  
#   return X, pool2


# def regression_head(x, conv_layer):
  
    
#   # Densely connected layer
#   W_fc1 = weight_variable([32 * 8 * 128, 3072])
#   b_fc1 = bias_variable([3072])

#   conv_layer_flat = tf.reshape(conv_layer, [BATCH_SIZE,-1])
#   bp()
#   h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

#   # Output layer
#   W_fc2 = weight_variable([3072, 10])
#   b_fc2 = bias_variable([10])

#   y = tf.matmul(h_fc1, W_fc2) + b_fc2
#   #bp()
#   return (x, y)


# def localization_head():
#   pass


# def data_type():
#   """Return the type of the activations, weights, and placeholder variables."""
#   return tf.float32


# # This is where training samples and labels are fed to the graph.
# # These placeholder nodes will be fed a batch of training data at each
# # training step using the {feed_dict} argument to the Run() call below.
# train_data_node = tf.placeholder(
#     data_type(),
#     shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
# train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
# eval_data = tf.placeholder(
#     data_type(),
#     shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

# # The variables below hold all the trainable weights. They are passed an
# # initial value which will be assigned when we call:
# # {tf.initialize_all_variables().run()}
# conv1_weights = tf.Variable(
#     tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
#                         stddev=0.1,
#                         seed=SEED, dtype=data_type()))
# conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
# conv2_weights = tf.Variable(tf.truncated_normal(
#     [5, 5, 32, 64], stddev=0.1,
#     seed=SEED, dtype=data_type()))
# conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
# fc1_weights = tf.Variable(  # fully connected, depth 512.
#     tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
#                         stddev=0.1,
#                         seed=SEED,
#                         dtype=data_type()))
# fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
# fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
#                                               stddev=0.1,
#                                               seed=SEED,
#                                               dtype=data_type()))
# fc2_biases = tf.Variable(tf.constant(
#     0.1, shape=[NUM_LABELS], dtype=data_type()))



# # We will replicate the model structure for the training subgraph, as well
# # as the evaluation subgraphs, while sharing the trainable parameters.
# def model(data, train=False):
#   """The Model definition."""
#   # 2D convolution, with 'SAME' padding (i.e. the output feature map has
#   # the same size as the input). Note that {strides} is a 4D array whose
#   # shape matches the data layout: [image index, y, x, depth].
#   conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
#   # Bias and rectified linear non-linearity.
#   relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
#   # Max pooling. The kernel size spec {ksize} also follows the layout of
#   # the data. Here we have a pooling window of 2, and a stride of 2.
#   pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#   conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
#   relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
#   pool = tf.nn.max_pool(relu,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME')
#   # Reshape the feature map cuboid into a 2D matrix to feed it to the
#   # fully connected layers.
#   pool_shape = pool.get_shape().as_list()
#   reshape = tf.reshape(
#       pool,
#       [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
#   # Fully connected layer. Note that the '+' operation automatically
#   # broadcasts the biases.
#   hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
#   # Add a 50% dropout during training only. Dropout also scales
#   # activations such that no rescaling is needed at evaluation time.
#   if train:
#     hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

#   return tf.matmul(hidden, fc2_weights) + fc2_biases