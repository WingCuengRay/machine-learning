import tensorflow as tf
from pdb import set_trace as bp
import numpy as np

NUM_CHANNELS = 3
NUM_LABELS = 11 # 0-9, + blank 
PATCH_SIZE = 5
DEPTH_1 = 16
DEPTH_2 = 20
DEPTH_3 =  20
DEPTH_4 = 512
num_hidden1 = 64
DROPOUT = 0.75

def _activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def convolution_model(data_node, batch_size, train):
  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_1], stddev=5e-2))
  conv1_biases =  tf.Variable(tf.zeros([DEPTH_1]))

  conv2_weights = tf.Variable(tf.truncated_normal(shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_1, DEPTH_2], stddev=5e-2))
  conv2_biases =  tf.Variable(tf.constant(0.1, shape=[DEPTH_2]))
  
  conv3_weights = tf.Variable(tf.truncated_normal(shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_2, DEPTH_3], stddev=5e-2))
  conv3_biases =  tf.Variable(tf.constant(0.1, shape=[DEPTH_3]))

  conv4_weights = tf.Variable(tf.truncated_normal(shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_3, DEPTH_4], stddev=5e-2))
  conv4_biases =  tf.Variable(tf.constant(0.1, shape=[DEPTH_4]))


  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  #if NUM_CHANNELS == 1: #If a greysacle image is sent expand the dimensions
  #  data_node = tf.expand_dims(data_node, 3)
  
  '''Layer 1'''
  with tf.variable_scope('conv_1') as scope:
    conv1 = tf.nn.conv2d(data_node, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    print("Conv 1 shape", conv1.get_shape().as_list())
    # Bias and rectified linear non-linearity.
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    print("Relu 1 shape", conv1.get_shape().as_list())
    _activation_summary(relu1)
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1,2,2,1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    print("Pool 1 shape", pool1.get_shape().as_list())

  '''Layer 2'''
  with tf.variable_scope('conv_2') as scope:
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
    print("Conv 2 shape", conv2.get_shape().as_list())
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    print("Relu 2 shape", relu2.get_shape().as_list())
    _activation_summary(relu2)
    norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
    print("Pool 2 shape", pool2.get_shape().as_list())

  '''Layer3'''
  with tf.variable_scope('conv_3') as scope:
    conv3 = tf.nn.conv2d(pool2, conv3_weights, strides = [1,1,1,1], padding='SAME')
    print("Conv 3 shape", conv3.get_shape().as_list())
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    print("Relu 3 shape", relu3.get_shape().as_list())
    _activation_summary(relu3)
    norm3 = tf.nn.lrn(relu3, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
    print("Pool 3 shape", pool3.get_shape().as_list())

  #'''Layer4'''
  #with tf.variable_scope('conv_4') as scope:
  #   conv4 = tf.nn.conv2d(pool3, conv4_weights, strides = [1,1,1,1], padding='SAME')
  #   relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
  #   _activation_summary(relu4)
  #   norm4 = tf.nn.lrn(relu4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
  #   pool4 = tf.nn.max_pool(norm4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool4')

  #'''Layer5'''
  # with tf.variable_scope('conv_4') as scope:
  #   conv5 = tf.nn.conv2d(pool3, conv4_weights, strides = [1,1,1,1], padding='SAME')
  #   relu5 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
  #   _activation_summary(relu4)
  #   norm4 = tf.nn.lrn(relu4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
  #   pool4 = tf.nn.max_pool(norm4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  weights = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, conv3_weights, conv3_biases]
  return pool3, weights



def classification_head(X, batch_size, train=False):
  conv_layer, conv_vars = convolution_model(X, batch_size, train=False)
  
  #Layer3
  dim = 4*4*20
  l3_weights = tf.Variable(tf.truncated_normal(shape=[dim, 10], stddev=0.04))
  l3_biases = tf.Variable(tf.constant(0.1, shape=[10]))

  out_weights = tf.Variable(tf.random_normal([10, 10]))
  out_biases =  tf.Variable(tf.constant(0.1, shape=[10]))

  #Fully Connected Layer
  with tf.variable_scope('fully_connected') as scope:
    fc1 = tf.reshape(conv_layer, [batch_size, -1])
    fc1 = tf.add(tf.matmul(fc1, l3_weights), l3_biases)
    fc_out = tf.nn.relu(fc1)
    print("fc shape", fc_out.get_shape().as_list())
    _activation_summary(fc_out)
 
  out = tf.add(tf.matmul(fc_out, out_weights), out_biases)

  #apply dopout to training.
  if train == True:
    print("using drop out")
    out = tf.nn.dropout(out, DROPOUT)
  else:
    print("not using dropout")

  weights = conv_vars + [l3_weights, l3_biases, out_weights, out_biases]

  #output class scores 
  return out, weights




layer1_weights = tf.get_variable("W1", shape=[PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH_1],\
         initializer=tf.contrib.layers.xavier_initializer_conv2d())
layer1_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_1]), name='B1')

layer2_weights = tf.get_variable("W2", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_1, DEPTH_2],\
         initializer=tf.contrib.layers.xavier_initializer_conv2d())
layer2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_2]), name='B2')

layer3_weights = tf.get_variable("W3", shape=[PATCH_SIZE, PATCH_SIZE, DEPTH_2, num_hidden1],\
         initializer=tf.contrib.layers.xavier_initializer_conv2d())
layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name='B3')

layer4_weights = tf.get_variable("W4", shape=[PATCH_SIZE, PATCH_SIZE, 64, 64],\
          initializer=tf.contrib.layers.xavier_initializer_conv2d())
layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name='B4')


s1_w = tf.get_variable("WS1", shape=[num_hidden1, NUM_LABELS],\
         initializer=tf.contrib.layers.xavier_initializer())
s1_b = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS1')

s2_w = tf.get_variable("WS2", shape=[num_hidden1, NUM_LABELS],\
         initializer=tf.contrib.layers.xavier_initializer())
s2_b = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS2')

s3_w = tf.get_variable("WS3", shape=[num_hidden1, NUM_LABELS],\
         initializer=tf.contrib.layers.xavier_initializer())
s3_b = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS3')

s4_w = tf.get_variable("WS4", shape=[num_hidden1, NUM_LABELS],\
         initializer=tf.contrib.layers.xavier_initializer())
s4_b = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS4')

s5_w = tf.get_variable("WS5", shape=[num_hidden1, NUM_LABELS],\
         initializer=tf.contrib.layers.xavier_initializer())
s5_b = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]), name='BS5')


def regression_head(X, batch_size, train=False):
  conv = tf.nn.conv2d(X, layer1_weights, [1,1,1,1], 'SAME', name='C1')
  print("Conv 1 shape", conv.get_shape().as_list())
  hidden = tf.nn.relu(conv + layer1_biases)
  lrn = tf.nn.local_response_normalization(hidden)
  sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S2')
  print("Layer 1 shape", sub.get_shape().as_list())

  conv = tf.nn.conv2d(sub, layer2_weights, [1,1,1,1], padding='SAME', name='C3')
  hidden = tf.nn.relu(conv + layer2_biases)
  lrn = tf.nn.local_response_normalization(hidden)
  sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S4')
  print("Layer 2 shape", sub.get_shape().as_list())  

  conv = tf.nn.conv2d(sub, layer3_weights, [1,1,1,1], padding='SAME', name='C5')
  hidden = tf.nn.relu(conv + layer3_biases)
  lrn = tf.nn.local_response_normalization(hidden)
  sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S5')
  print("Layer 3 shape", sub.get_shape().as_list())
  
  #bp()
  # conv = tf.nn.conv2d(hidden, layer4_weights, [1,1,1,1], padding='SAME', name='C6')
  # hidden = tf.nn.relu(conv + layer4_biases)
  # lrn = tf.nn.local_response_normalization(hidden)
  # sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S6')
  # print("Layer 4 shape", sub.get_shape().as_list())


  #bp()


  shape = hidden.get_shape().as_list()
  reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    
  #hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
  logits1 = tf.matmul(reshape, s1_w) + s1_b
  logits2 = tf.matmul(reshape, s2_w) + s2_b
  logits3 = tf.matmul(reshape, s3_w) + s3_b
  logits4 = tf.matmul(reshape, s4_w) + s4_b
  logits5 = tf.matmul(reshape, s5_w) + s5_b
 
  return [logits1, logits2, logits3, logits4, logits5]



