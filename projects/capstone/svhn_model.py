import tensorflow as tf

LEARNING_RATE = .001


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

#Utilites
def weight_variable(shape):
	"""Create a weight variable with appropriate initialization."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	"""Create a bias variable with appropriate initialization."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

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

def accuracy(y, y_):
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.scalar_summary('accuracy', accuracy)
	return(accuracy)

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


def model(X, y_, keep_prob):
  #create a 2-layer model with dropout
  hidden1 = nn_layer(X, 3072, 500, 'layer1')
  dropped = dropout(hidden1, keep_prob)

  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)
  cross_ent = cross_entropy(y, y_)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_ent)

  acc = accuracy(y, y_)  
  return acc, train_step


  def regression_head():
  	pass

  def localization_head():
  	pass