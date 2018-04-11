import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class VGG:
    """
    VGG class used to load VGG 16 and 19 architectures from paper:
    https://arxiv.org/pdf/1409.1556.pdf
    """

    def __init__(self, weights_npy_file):
        #TODO
        print("TODO")

    def conv_layer(self, input, filter, strides=[1,1,1,1], kernel=[1,3,3,1], padding='SAME', name):
        # Get input shape
        input_shape = input.get_shape().as_list()

        # Define weights and bias
        weights = tf.Variable(tf.truncated_normal(shape=[input_shape[1], filter], mean=0, stddev=0.1))
        bias = tf.Variable(tf.zeros(filter))

        # Apply convolution layer
        conv = tf.nn.conv2d(input, filter, strides, padding, name)

        # Add bias
        conv = tf.nn.bias_add(conv, bias)

        # Apply activation function
        conv = tf.nn.relu(conv)
        return conv

    def max_pool_layer(self, input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name):
        return tf.nn.max_pool(input, ksize, strides, padding, name)

    def dense_layer(self, input, neurons, name):
        # Get input shape
        input_shape = input.get_shape().as_list()

        # Define weights and bias
        weights = tf.Variable(tf.truncated_normal(shape=[input_shape[1], neurons], mean=0, stddev=0.1))
        bias = tf.Variable(tf.zeros(filter))

        # Apply matrix multiplication
        dense = tf.add(tf.matmul(input, weights), bias)

        # Apply activation function
        dense = tf.nn.relu(dens)
        return dense

    def vgg16(self, features):
        """
        Builds VGG16
        """

        # Block 1
        conv1_1 = self.conv_layer(features, 64, 'conv1_1')
        conv1_2 = self.conv_layer(conv1_1, 64, 'conv1_2')
        pool1 = self.max_pool_layer(conv1_2, 'pool1')
        # Block 2
        conv2_1 = self.conv_layer(pool1, 128, 'conv2_1')
        conv2_2 = self.conv_layer(conv2_1, 128, 'conv2_2')
        pool2 = self.max_pool_layer(conv2_2, 'pool2')
        # Block 3
        conv3_1 = self.conv_layer(pool2, 256, 'conv3_1')
        conv3_2 = self.conv_layer(conv3_1, 256, 'conv3_2')
        conv3_3 = self.conv_layer(conv3_2, 256, 'conv3_3')
        pool3 = self.max_pool_layer(conv3_3, 'pool3')
        # Block 4
        conv4_1 = self.conv_layer(pool3, 256, 'conv4_1')
        conv4_2 = self.conv_layer(conv4_1, 256, 'conv4_2')
        conv4_3 = self.conv_layer(conv4_2, 256, 'conv4_3')
        pool4 = self.max_pool_layer(conv4_3, 'pool4')
        # Block 5
        conv5_1 = self.conv_layer(pool4, 256, 'conv3_1')
        conv5_2 = self.conv_layer(conv5_1, 256, 'conv3_2')
        conv5_3 = self.conv_layer(conv5_2, 256, 'conv3_3')
        pool5 = self.max_pool_layer(conv5_3, 'pool3')

        # Flatten layer
        flatten_layer = flatten(pool5)

        # Dense 1
        fc6 = self.dense_layer(flatten_layer, 4096, 'fc6')
        # Dense 2
        fc7 = self.dense_layer(fc6, 4096, 'fc7')
        # Dense 3
        fc8 = self.dense_layer(fc7, 1000, 'fc8')

        # Softmax
        prob = tf.nn.softmax(fc8, name='prob')
        return prob

    def vgg19(self, features):
        """
        Builds VGG19
        """
        return None
