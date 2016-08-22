import tensorflow as tf
from pdb import set_trace as bp
import numpy as np

NUM_CHANNELS = 3
NUM_LABELS = 11 # 0-9, + blank 

#Hyper Parameters
PATCH_SIZE = 5
DEPTH_1 = 16
DEPTH_2 = 32
DEPTH_3 =  64
num_hidden1 = 128
DROPOUT = 0.85


#Convolution Weight and Bias Variables 
conv1_weights = tf.get_variable("W1", shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_1])
conv1_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_1]), name='B1')

conv2_weights = tf.get_variable("W2", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_1, DEPTH_2])
conv2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_2]), name='B2')

conv3_weights = tf.get_variable("W3", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_2, DEPTH_3])
conv3_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_3]), name='B3')

conv4_weights = tf.get_variable("W4", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_3, num_hidden1])
conv4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name='B4')

# Regression Weight and Bias Variables
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

#Classification Weight and Bias Variables

cl_l3_weights = tf.get_variable("Classifer_Weights_1", shape=[64, 64], initializer=tf.contrib.layers.xavier_initializer())
cl_l3_biases = tf.Variable(tf.constant(0.0, shape=[64]), name='Classifer_Biases_1')

# cl_l4_weights = tf.get_variable("Classifer_Weights_2", shape=[384, 192], initializer=tf.contrib.layers.xavier_initializer())
# cl_l4_biases = tf.Variable(tf.constant(0.0, shape=[192]), name='Classifer_Biases_2')

cl_out_weights = tf.get_variable("Classifer_Weights_3", shape=[64, 10], initializer=tf.contrib.layers.xavier_initializer())
cl_out_biases = tf.Variable(tf.constant(0.0, shape=[10]), name='Classifer_Biases_3')


def _activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def convolution_model(data):
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
    #if(regression)
    #sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S5')
    #if(classification)
    sub = tf.nn.max_pool(lrn, [1,1,1,1], [1,1,1,1], 'SAME', name='S5')
    _activation_summary(sub)
  
  return sub


def classification_head(data, keep_prob=1.0, train=False):
  conv_layer = convolution_model(data)
  shape = conv_layer.get_shape().as_list()
  dim = shape[1] * shape[2] * shape[3]

  if train == True:
    print("using drop out")
    fc_out = tf.nn.dropout(conv_layer, DROPOUT)
  else:
    print("not using dropout")

  #Fully Connected Layer 1
  with tf.variable_scope('fully_connected_1') as scope:
    fc1 = tf.reshape(conv_layer, [shape[0], -1])
    fc1 = tf.add(tf.matmul(fc1, cl_l3_weights), cl_l3_biases)
    fc_out = tf.nn.relu(fc1, name=scope.name)
    _activation_summary(fc_out)
     
  # Fully Connected Layer 2
  # with tf.variable_scope('fully_connected_2') as scope:
  #   fc2 = tf.add(tf.matmul(fc_out, cl_l4_weights), cl_l4_biases)
  #   fc2_out = tf.nn.relu(fc2, name=scope.name)
  #   _activation_summary(fc2_out)

  with tf.variable_scope("softmax_linear") as scope:
    logits = tf.matmul(fc_out, cl_out_weights) + cl_out_biases
    _activation_summary(logits)

  #Output class scores
  return logits


def regression_head(data, keep_prob=1.0):
  conv_layer = convolution_model(data)

  with tf.variable_scope('full_connected_1') as scope:
    conv = tf.nn.conv2d(conv_layer, conv4_weights, [1,1,1,1], padding='VALID', name='C5')
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