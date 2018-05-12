import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class VGG:
    """
    VGG class used to load VGG16 and VGG19 architectures from paper:
    https://arxiv.org/pdf/1409.1556.pdf

    Pre-trained weights are loaded from Numpy (npy) files.
    Numpy files have been downloaded from:
    https://github.com/machrisaa/tensorflow-vgg
    """

    def __init__(self, vgg16_weights_file=None, vgg19_weights_file=None):
        self.vgg16_weights = None
        self.vgg19_weights = None

        # Load weights for VGG16
        if vgg16_weights_file is not None:
            t1 = time.time()
            self.vgg16_weights = np.load(vgg16_weights_file, encoding='latin1').item()
            t2 = time.time()
            print("Pre-trained weights for VGG16 loaded in {0:.2f}s".format(t2-t1))

        # Load weights for VGG19
        if vgg19_weights_file is not None:
            t1 = time.time()
            self.vgg19_weights = np.load(vgg19_weights_file, encoding='latin1').item()
            t2 = time.time()
            print("Pre-trained weights for VGG19 loaded in {0:.2f}s".format(t2-t1))

    def conv_layer(self, input, strides=[1,1,1,1], padding='SAME', name='conv', model='vgg16'):
        with tf.name_scope(name):
            # Define weights and bias
            weights = self.get_weights(name, model)
            bias = self.get_bias(name, model)

            # Apply convolution layer
            conv = tf.nn.conv2d(input, weights, strides, padding, name=name)

            # Add bias
            conv = tf.nn.bias_add(conv, bias)

            # Apply activation function
            return tf.nn.relu(conv)

    def max_pool_layer(self, input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool'):
        with tf.name_scope(name):
            return tf.nn.max_pool(input, ksize, strides, padding, name=name)

    def dense_layer(self, input, name='dense', model='vgg16'):
        with tf.name_scope(name):
            # Define weights and bias
            weights = self.get_weights(name, model)
            bias = self.get_bias(name, model)

            # Apply matrix multiplication
            dense = tf.add(tf.matmul(input, weights), bias)

            # Apply activation function
            return tf.nn.relu(dense)

    def get_weights(self, name, model):
        if model == 'vgg19':
            return tf.constant(self.vgg19_weights[name][0])
        else:
            return tf.constant(self.vgg16_weights[name][0])

    def get_bias(self, name, model):
        if model == 'vgg19':
            return tf.constant(self.vgg19_weights[name][1])
        else:
            return tf.constant(self.vgg16_weights[name][1])

    def vgg16(self, features):
        """
        Builds VGG16
        """

        with tf.name_scope('vgg16'):
            # Store network layers
            layers = {}

            # Block 1
            conv1_1 = self.conv_layer(features, name='conv1_1')
            conv1_2 = self.conv_layer(conv1_1, name='conv1_2')
            pool1 = self.max_pool_layer(conv1_2, name='pool1')
            layers['conv1_1'] = conv1_1
            layers['conv1_2'] = conv1_2
            layers['pool1'] = pool1
            # Block 2
            conv2_1 = self.conv_layer(pool1, name='conv2_1')
            conv2_2 = self.conv_layer(conv2_1, name='conv2_2')
            pool2 = self.max_pool_layer(conv2_2, name='pool2')
            layers['conv2_1'] = conv2_1
            layers['conv2_2'] = conv2_2
            layers['pool2'] = pool2
            # Block 3
            conv3_1 = self.conv_layer(pool2, name='conv3_1')
            conv3_2 = self.conv_layer(conv3_1, name='conv3_2')
            conv3_3 = self.conv_layer(conv3_2, name='conv3_3')
            pool3 = self.max_pool_layer(conv3_3, name='pool3')
            layers['conv3_1'] = conv3_1
            layers['conv3_2'] = conv3_2
            layers['conv3_3'] = conv3_3
            layers['pool3'] = pool3
            # Block 4
            conv4_1 = self.conv_layer(pool3, name='conv4_1')
            conv4_2 = self.conv_layer(conv4_1, name='conv4_2')
            conv4_3 = self.conv_layer(conv4_2, name='conv4_3')
            pool4 = self.max_pool_layer(conv4_3, name='pool4')
            layers['conv4_1'] = conv4_1
            layers['conv4_2'] = conv4_2
            layers['conv4_3'] = conv4_3
            layers['pool4'] = pool4
            # Block 5
            conv5_1 = self.conv_layer(pool4, name='conv5_1')
            conv5_2 = self.conv_layer(conv5_1, name='conv5_2')
            conv5_3 = self.conv_layer(conv5_2, name='conv5_3')
            pool5 = self.max_pool_layer(conv5_3, name='pool5')
            layers['conv5_1'] = conv5_1
            layers['conv5_2'] = conv5_2
            layers['conv5_3'] = conv5_3
            layers['pool5'] = pool5

            '''
            # Flatten layer
            flatten_layer = flatten(pool5)

            # Dense 1
            fc6 = self.dense_layer(flatten_layer, name='fc6')
            layers['fc6'] = fc6
            # Dense 2
            fc7 = self.dense_layer(fc6, name='fc7')
            layers['fc7'] = fc7
            # Dense 3
            fc8 = self.dense_layer(fc7, name='fc8')
            layers['fc8'] = fc8

            # Softmax
            prob = tf.nn.softmax(fc8, name='prob')'''

            # Return network layers
            return layers

    def vgg19(self, features):
        """
        Builds VGG19
        """

        with tf.name_scope('vgg19'):
            layers = {}

            # Block 1
            conv1_1 = self.conv_layer(features, name='conv1_1', model='vgg19')
            conv1_2 = self.conv_layer(conv1_1, name='conv1_2', model='vgg19')
            pool1 = self.max_pool_layer(conv1_2, name='pool1')
            layers['conv1_1'] = conv1_1
            layers['conv1_2'] = conv1_2
            layers['pool1'] = pool1
            # Block 2
            conv2_1 = self.conv_layer(pool1, name='conv2_1', model='vgg19')
            conv2_2 = self.conv_layer(conv2_1, name='conv2_2', model='vgg19')
            pool2 = self.max_pool_layer(conv2_2, name='pool2')
            layers['conv2_1'] = conv2_1
            layers['conv2_2'] = conv2_2
            layers['pool2'] = pool2
            # Block 3
            conv3_1 = self.conv_layer(pool2, name='conv3_1', model='vgg19')
            conv3_2 = self.conv_layer(conv3_1, name='conv3_2', model='vgg19')
            conv3_3 = self.conv_layer(conv3_2, name='conv3_3', model='vgg19')
            conv3_4 = self.conv_layer(conv3_3, name='conv3_4', model='vgg19')
            pool3 = self.max_pool_layer(conv3_4, name='pool3')
            layers['conv3_1'] = conv3_1
            layers['conv3_2'] = conv3_2
            layers['conv3_3'] = conv3_3
            layers['conv3_4'] = conv3_4
            layers['pool3'] = pool3
            # Block 4
            conv4_1 = self.conv_layer(pool3, name='conv4_1', model='vgg19')
            conv4_2 = self.conv_layer(conv4_1, name='conv4_2', model='vgg19')
            conv4_3 = self.conv_layer(conv4_2, name='conv4_3', model='vgg19')
            conv4_4 = self.conv_layer(conv4_3, name='conv4_4', model='vgg19')
            pool4 = self.max_pool_layer(conv4_4, name='pool4')
            layers['conv4_1'] = conv4_1
            layers['conv4_2'] = conv4_2
            layers['conv4_3'] = conv4_3
            layers['conv4_4'] = conv4_4
            layers['pool4'] = pool4
            # Block 5
            conv5_1 = self.conv_layer(pool4, name='conv5_1', model='vgg19')
            conv5_2 = self.conv_layer(conv5_1, name='conv5_2', model='vgg19')
            conv5_3 = self.conv_layer(conv5_2, name='conv5_3', model='vgg19')
            conv5_4 = self.conv_layer(conv5_3, name='conv5_4', model='vgg19')
            pool5 = self.max_pool_layer(conv5_4, name='pool5')
            layers['conv5_1'] = conv5_1
            layers['conv5_2'] = conv5_2
            layers['conv5_3'] = conv5_3
            layers['conv5_4'] = conv5_4
            layers['pool5'] = pool5

            '''
            # Flatten layer
            flatten_layer = flatten(pool5)

            # Dense 1
            fc6 = self.dense_layer(flatten_layer, name='fc6', model='vgg19')
            layers['fc6'] = fc6
            # Dense 2
            fc7 = self.dense_layer(fc6, name='fc7', model='vgg19')
            layers['fc7'] = fc7
            # Dense 3
            fc8 = self.dense_layer(fc7, name='fc8', model='vgg19')
            layers['fc8'] = fc8

            # Softmax
            prob = tf.nn.softmax(fc8, name='prob')'''

            # Return network layers
            return layers
