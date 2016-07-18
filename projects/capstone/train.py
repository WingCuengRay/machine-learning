import cv2
import numpy as np
import tensorflow as tf

from data import load_minst_data

img_rows = 28
img_cols = 28
num_labels = 10
num_channels = 1 # grayscale
img_cols = 28
img_rows = 28
patch_size = 5
depth = 32
num_hidden = 64
batch_size = 16
num_steps = 2000
SEED = None

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, img_rows, img_cols, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

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


def train(graph, train_dataset, train_labels, valid_dataset, 
		valid_labels, test_dataset, test_labels):
	with graph.as_default():
		# Input data.
		tf_train_dataset = tf.placeholder(
			tf.float32, shape=(batch_size, img_rows, img_cols, num_channels))
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)

		# Variables.
		layer1_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, num_channels, depth], stddev=0.1))
		layer1_biases = tf.Variable(tf.zeros([depth]))

		layer2_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, depth, depth], stddev=0.1))
		layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

		fc1_weights = tf.Variable(tf.truncated_normal(
		 	[img_rows // 4 * img_cols // 4 * depth, num_hidden], stddev=0.1))
		fc1_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

		fc2_weights = tf.Variable(tf.truncated_normal(
			[num_hidden, num_labels], stddev=0.1))
		fc2_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

		# Model.
		def model(data, train=False):
			"""The Model definition."""
			# Bias and rectified linear non-linearity.
			
			'''First Convo Layer.'''
			conv1 = tf.nn.conv2d(data, layer1_weights, strides=[1, 1, 1, 1], padding='SAME')
			relu1 = tf.nn.relu(tf.nn.bias_add(conv1, layer1_biases))
			pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			
			'''Second Convo Layer.'''
			conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
			relu2 = tf.nn.relu(tf.nn.bias_add(conv2, layer2_biases))
			pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			
			'''Densely Connected Layer.'''
			shape = pool2.get_shape().as_list()
			reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
			hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
			
			if train:
				hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

			return tf.matmul(hidden, fc2_weights) + fc2_biases

		# Training computation.
		logits = model(tf_train_dataset, True)
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
	#train(graph, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

if __name__ == '__main__':
	train_dataset, train_labels = load_minst_data("training")
	test_dataset, test_labels = load_minst_data("testing")
	valid_dataset, valid_labels = load_minst_data("validation")
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)
	t_p(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)