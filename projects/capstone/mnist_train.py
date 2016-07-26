import numpy as np
import tensorflow as tf
from mnist_data import load_mnist_data
#from pdb import set_trace as bp

img_rows = 28
img_cols = 28
NUM_LABELS = 10
NUM_CHANNELS = 1 # grayscale
patch_size = 5
depth = 32
num_hidden = 64
batch_size = 64
num_steps = 10000
SEED = None
data_type = tf.float32

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
		/ predictions.shape[0])

def predict(graph, train_dataset, valid_prediction, test_prediction, 
		train_labels, valid_labels, test_labels, tf_train_dataset,
		tf_train_labels, optimizer, loss, train_prediction):
	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		print('Initialized')
		for step in range(num_steps):
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
			batch_labels = train_labels[offset:(offset + batch_size), :]
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			_, l, predictions = session.run(
				[optimizer, loss, train_prediction], feed_dict=feed_dict)
			if (step % 100 == 0):
				print('Minibatch loss at step %d: %f' % (step, l))
				print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
				print('Validation accuracy: %.1f%%' % accuracy(
					valid_prediction.eval(), valid_labels))
		print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


def train(graph, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
	with graph.as_default():
		# Input data.
		tf_train_dataset = tf.placeholder(
			tf.float32, shape=(batch_size, img_rows, img_cols, NUM_CHANNELS))
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)
	  
		# # Variables.
		# layer1_weights = tf.Variable(tf.truncated_normal(
		# 	[patch_size, patch_size, num_channels, depth], stddev=0.1))
		# layer1_biases = tf.Variable(tf.zeros([depth]))
		# layer2_weights = tf.Variable(tf.truncated_normal(
		# 	[patch_size, patch_size, depth, depth], stddev=0.1))
		# layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
		# layer3_weights = tf.Variable(tf.truncated_normal(
		# 	[img_rows // 4 * img_cols // 4 * depth, num_hidden], stddev=0.1))
		# layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
		# layer4_weights = tf.Variable(tf.truncated_normal(
		# 	[num_hidden, num_labels], stddev=0.1))
		# layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
	  
		# # Model.
		# def model(data):
		# 	conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
		# 	hidden = tf.nn.relu(conv + layer1_biases)
		# 	conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
		# 	hidden = tf.nn.relu(conv + layer2_biases)
		# 	shape = hidden.get_shape().as_list()
		# 	reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		# 	hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
		# 	return tf.matmul(hidden, layer4_weights) + layer4_biases
############################################
  		# The variables below hold all the trainable weights. They are passed an
  		# initial value which will be assigned when we call:
  		# {tf.initialize_all_variables().run()}
		#Layer1
		conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=data_type))
		conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type))
  		
  		#Layer 2
		conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type))
		conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type))
		#Layer 3
		fc1_weights = tf.Variable(  # fully connected, depth 512.
		tf.truncated_normal([img_rows // 4 * img_cols // 4 * 64, 512],
			stddev=0.1,
			seed=SEED,
			dtype=data_type))
		fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type))
		#Layer 4
		fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
			stddev=0.1,
			seed=SEED,
			dtype=data_type))
		fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type))

		# We will replicate the model structure for the training subgraph, as well
		# as the evaluation subgraphs, while sharing the trainable parameters.
		def model(data, train=False):
			"""The Model definition."""
			# 2D convolution, with 'SAME' padding (i.e. the output feature map has
			# the same size as the input). Note that {strides} is a 4D array whose
	 		# shape matches the data layout: [image index, y, x, depth].
			conv = tf.nn.conv2d(data,
				conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
			# Bias and rectified linear non-linearity.
			relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
			# Max pooling. The kernel size spec {ksize} also follows the layout of
			# the data. Here we have a pooling window of 2, and a stride of 2.
			pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='SAME')
			conv = tf.nn.conv2d(pool, conv2_weights,
				strides=[1, 1, 1, 1], padding='SAME')
			relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
			pool = tf.nn.max_pool(relu,
				ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1],
				padding='SAME')
			# Reshape the feature map cuboid into a 2D matrix to feed it to the
			# fully connected layers.
			pool_shape = pool.get_shape().as_list()
			reshape = tf.reshape( pool,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
			# Fully connected layer. Note that the '+' operation automatically
			# broadcasts the biases.
			hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
			# Add a 50% dropout during training only. Dropout also scales
			# activations such that no rescaling is needed at evaluation time.
			if train:
				hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
			return tf.matmul(hidden, fc2_weights) + fc2_biases
############################################
		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	
		# Optimizer.
		optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)	
	
		'''Predictions for the training, validation, and test data.'''
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
		test_prediction = tf.nn.softmax(model(tf_test_dataset))
		return train_prediction, valid_prediction, test_prediction, tf_train_labels, optimizer, loss, tf_train_dataset

def t_p(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
	graph = tf.Graph()
	train_prediction, valid_prediction, test_prediction, tf_train_labels, optimizer, loss, tf_train_dataset = train(
		graph, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
	predict(graph, train_dataset, valid_prediction, test_prediction, train_labels, valid_labels, 
		test_labels, tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction)

if __name__ == '__main__':
	train_dataset, train_labels = load_mnist_data("training")
	test_dataset, test_labels = load_mnist_data("testing")
	valid_dataset, valid_labels = load_mnist_data("validation")
	t_p(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)