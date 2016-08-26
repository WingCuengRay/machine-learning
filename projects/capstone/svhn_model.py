import tensorflow as tf

# Image Parameters
NUM_CHANNELS = 3
CL_NUM_LABELS = 10
NUM_LABELS = CL_NUM_LABELS + 1  # 0-9, + 1 blank


# Hyper Parameters
PATCH_SIZE = 5
DEPTH_1 = 16
DEPTH_2 = 32
DEPTH_3 = 64
num_hidden1 = 128
DROPOUT = 0.85


# Convolution Weight and Bias Variables
conv1_weights = tf.get_variable("Weights_1", shape=[PATCH_SIZE, PATCH_SIZE,
                                NUM_CHANNELS, DEPTH_1])
conv1_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_1]), name='Biases_1')

conv2_weights = tf.get_variable("Weights_2", shape=[PATCH_SIZE, PATCH_SIZE,
                                DEPTH_1, DEPTH_2])
conv2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_2]), name='Biases_2')

conv3_weights = tf.get_variable("Weights_3", shape=[PATCH_SIZE, PATCH_SIZE,
                                DEPTH_2, DEPTH_3])
conv3_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH_3]), name='Biases_3')


conv4_weights = tf.get_variable("Weights_4", shape=[PATCH_SIZE,
                                PATCH_SIZE, DEPTH_3, num_hidden1])
conv4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name='Biases_4')

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

# Classification Weight and Bias Variables

cl_l3_weights = tf.get_variable("Classifer_Weights_1", shape=[64, 64])
cl_l3_biases = tf.Variable(tf.constant(0.0, shape=[64]),
                           name='Classifer_Biases_1')

# cl_l4_weights = tf.get_variable("Classifer_Weights_2", shape=[384, 192])
# cl_l4_biases = tf.Variable(tf.constant(0.0, shape=[192]),
#                            name='Classifer_Biases_2')

cl_out_weights = tf.get_variable("Classifer_Weights_3",
                                 shape=[64, CL_NUM_LABELS])
cl_out_biases = tf.Variable(tf.constant(0.0, shape=[CL_NUM_LABELS]),
                            name='Classifer_Biases_3')


def activation_summary(x):
    tensor_name = x.op.name
    # tf.histogram_summary(tensor_name + '/activations', x)
    # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def convolution_model(data):
    with tf.variable_scope('conv_1', reuse=True) as scope:
        con = tf.nn.conv2d(data, conv1_weights,
                           [1, 1, 1, 1], 'VALID', name='C1')
        hid = tf.nn.relu(con + conv1_biases)
        lrn = tf.nn.local_response_normalization(hid)
        sub = tf.nn.max_pool(lrn,
                             [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='S2')
        activation_summary(sub)

    with tf.variable_scope('conv_2') as scope:
        con = tf.nn.conv2d(sub, conv2_weights,
                           [1, 1, 1, 1], padding='VALID', name='C3')
        hid = tf.nn.relu(con + conv2_biases)
        lrn = tf.nn.local_response_normalization(hid)
        sub = tf.nn.max_pool(lrn,
                             [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='S4')
        activation_summary(sub)

    with tf.variable_scope('conv_3') as scope:
        con = tf.nn.conv2d(sub, conv3_weights,
                           [1, 1, 1, 1], padding='VALID', name='C5')
        hid = tf.nn.relu(con + conv3_biases)
        lrn = tf.nn.local_response_normalization(hid)
        if lrn.get_shape().as_list()[1] is 1:  # Is already reduced.
            sub = tf.nn.max_pool(lrn,
                                 [1, 1, 1, 1], [1, 1, 1, 1], 'SAME', name='S5')
        else:
            sub = tf.nn.max_pool(lrn,
                                 [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='S5')

        activation_summary(sub)

    return sub


def classification_head(data, keep_prob=1.0, train=False):
    conv_layer = convolution_model(data)
    shape = conv_layer.get_shape().as_list()
    dim = shape[1] * shape[2] * shape[3]

    # with tf.name_scope('dropout'):
    if train is True:
        print("Using drop out")
        fc_out = tf.nn.dropout(conv_layer, DROPOUT)
    else:
        print("Not using dropout")
        #tf.scalar_summary('dropout_keep_probability', DROPOUT)

    # Fully Connected Layer 1
    with tf.variable_scope('fully_connected_1') as scope:
        fc1 = tf.reshape(conv_layer, [shape[0], -1])
        fc1 = tf.add(tf.matmul(fc1, cl_l3_weights), cl_l3_biases)
        fc_out = tf.nn.relu(fc1, name=scope.name)
        activation_summary(fc_out)

    # Fully Connected Layer 2
    # with tf.variable_scope('fully_connected_2') as scope:
    #   fc2 = tf.add(tf.matmul(fc_out, cl_l4_weights), cl_l4_biases)
    #   fc2_out = tf.nn.relu(fc2, name=scope.name)
    #   _activation_summary(fc2_out)

    with tf.variable_scope("softmax_linear") as scope:
        logits = tf.matmul(fc_out, cl_out_weights) + cl_out_biases
        activation_summary(logits)

    # Output class scores
    return logits


def regression_head(data, keep_prob=1.0):
    conv_layer = convolution_model(data)

    with tf.variable_scope('full_connected_1') as scope:
        conv = tf.nn.conv2d(conv_layer, conv4_weights, [1, 1, 1, 1],
                            padding='VALID', name='C5')
        hidden = tf.nn.relu(conv + conv4_biases)
        hidden = tf.nn.dropout(hidden, keep_prob)
        activation_summary(hidden)

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.variable_scope('Output') as scope:
        logits_1 = tf.matmul(reshape, reg1_weights) + reg1_biases
        logits_2 = tf.matmul(reshape, reg2_weights) + reg2_biases
        logits_3 = tf.matmul(reshape, reg3_weights) + reg3_biases
        logits_4 = tf.matmul(reshape, reg4_weights) + reg4_biases
        logits_5 = tf.matmul(reshape, reg5_weights) + reg5_biases

    return [logits_1, logits_2, logits_3, logits_4, logits_5]
