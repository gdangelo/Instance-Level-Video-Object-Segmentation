import time
import numpy as np
import tensorflow as tf
from vgg import VGG

class FCN:
    """
    FCN class
    """

    def conv_layer_1x1(self, input, out_channels, name=None):
        # Define weights and bias
        weights = tf.Variable(tf.truncated_normal(
            shape=[1, 1, input.get_shape().as_list()[3], out_channels],
            mean=0,
            stddev=0.1))
        bias = tf.Variable(tf.zeros(out_channels))

        # Apply convolution and add bias
        conv = tf.nn.conv2d(input, weights, [1,1,1,1], 'VALID', name=name)
        conv = tf.nn.bias_add(conv, bias)

        # Apply ReLU
        conv = tf.nn.relu(conv)

        return conv

    def back_conv_layer(self, input, factor, num_classes, name=None):
        # Get input shape
        input_shape = input.get_shape().as_list()
        batch_size = tf.shape(input)[0]

        # Get output shape
        new_height = input_shape[1] * factor
        new_width = input_shape[2] * factor
        output_shape = [batch_size, new_height, new_width, num_classes]

        # Get kernel size
        ksize = 2 * factor - factor % 2

        # Defines weights
        weights = tf.Variable(tf.truncated_normal(
            shape=[ksize, ksize, num_classes, input_shape[3]],
            mean=0,
            stddev=0.1))

        # Apply backward strided convolution
        back_conv = tf.nn.conv2d_transpose(
            input,
            weights,
            output_shape,
            strides=[1,factor,factor,1],
            padding='SAME',
            name=name)

        return back_conv

    def fcn_vgg_16(self, weights, features):
        # Build vgg16
        vgg = VGG(weights, None)
        vgg_16_logits, vgg_16_layers = vgg.vgg16(features)

        # Replace dense layers with 1x1 convolution layers
        pool5 = vgg_16_layers['pool5']
        conv6_1 = self.conv_layer_1x1(pool5, 4096, 'conv6_1')
        conv7_1 = self.conv_layer_1x1(conv6_1, 4096, 'conv7_1')

        # Upsampled to output segmentation using backward stride convolutions
        fcn32 = self.back_conv_layer(conv7_1, 32, 7)

        # Return heatmap predictions
        return tf.argmax(fcn32, axis=3)
