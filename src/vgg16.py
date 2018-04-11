import numpy as np
import tensorflow as tf

def VGG16(features):
    """
    Builds VGG16
    """

    # Convolutional Layer 1: 64 3X3 filters, stride 1, same padding, ReLU
    conv1 = tf.layers.conv2d(
        inputs=features,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv1')
    # Convolutional Layer 2: 64 3X3 filters, stride 1, same padding, ReLU
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv2')
    # Pooling Layer 1: Max-pooling with 2x2 filter, stride 2
    pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name='pool1')

    # ---

    # Convolutional Layer 3: 128 3X3 filters, stride 1, same padding, ReLU
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv3')
    # Convolutional Layer 4: 128 3X3 filters, stride 1, same padding, ReLU
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv4')
    # Pooling Layer 2: Max-pooling with 2x2 filter, stride 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv4,
        pool_size=[2, 2],
        strides=2,
        pool='pool2')

    # ---

    # Convolutional Layer 5: 256 3X3 filters, stride 1, same padding, ReLU
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv5')
    # Convolutional Layer 6: 256 3X3 filters, stride 1, same padding, ReLU
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv6')
    # Convolutional Layer 7: 256 3X3 filters, stride 1, same padding, ReLU
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv7')
    # Pooling Layer 3: Max-pooling with 2x2 filter, stride 2
    pool3 = tf.layers.max_pooling2d(
        inputs=conv7,
        pool_size=[2, 2],
        strides=2,
        name='pool3')

    # ---

    # Convolutional Layer 8: 512 3X3 filters, stride 1, same padding, ReLU
    conv8 = tf.layers.conv2d(
        inputs=conv7,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv8')
    # Convolutional Layer 9: 512 3X3 filters, stride 1, same padding, ReLU
    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv9')
    # Convolutional Layer 10: 512 3X3 filters, stride 1, same padding, ReLU
    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv10')
    # Pooling Layer 4: Max-pooling with 2x2 filter, stride 2
    pool4 = tf.layers.max_pooling2d(
        inputs=conv10,
        pool_size=[2, 2],
        strides=2,
        name='pool4')

    # ---

    # Convolutional Layer 11: 512 3X3 filters, stride 1, same padding, ReLU
    conv11 = tf.layers.conv2d(
        inputs=conv10,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv11')
    # Convolutional Layer 12: 512 3X3 filters, stride 1, same padding, ReLU
    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv12')
    # Convolutional Layer 13: 512 3X3 filters, stride 1, same padding, ReLU
    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        name='conv13')
    # Pooling Layer 5: Max-pooling with 2x2 filter, stride 2
    pool5 = tf.layers.max_pooling2d(
        inputs=conv13,
        pool_size=[2, 2],
        strides=2,
        name='pool5')
    # ---

    flatten = tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))])

    # Dense Layer 1: 4,096 neurons
    dense1 = tf.layers.dense(
        inputs=flatten,
        units=4096,
        activation=tf.nn.relu,
        name='dense1')
    # Dense Layer 2: 4,096 neurons
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=4096,
        activation=tf.nn.relu,
        name='dense2')
    # Dense Layer 3: 1,000 neurons
    logits = tf.layers.dense(
        inputs=dense2,
        units=1000,
        activation=tf.nn.relu,
        name='logits')

    # ---

    # Softmax
    probabilities = tf.nn.softmax(logits)

    return probabilities
