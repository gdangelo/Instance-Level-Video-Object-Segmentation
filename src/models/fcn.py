import time
import numpy as np
import tensorflow as tf
from models.vgg import VGG

class FCN:
    """
    FCN class
    """

    def conv_layer_1x1(self, input, out_channels, name=None):
        # Define weights and bias
        weights = tf.Variable(tf.truncated_normal(
            shape=[1, 1, input.get_shape().as_list()[3], out_channels],
            mean=0,
            stddev=0.1),
            name='weights_'+str(name))
        bias = tf.Variable(tf.zeros(out_channels), name='bias_'+str(name))

        # Apply convolution and add bias
        conv = tf.nn.conv2d(input, weights, [1,1,1,1], 'SAME', name=name)
        conv = tf.nn.bias_add(conv, bias)

        # Apply ReLU
        conv = tf.nn.relu(conv)

        return conv

    def back_conv_layer(self, input, factor, num_classes, name=None):
        # Get input shape
        input_shape = tf.shape(input)
        in_features = input.get_shape()[3].value

        # Get output shape
        new_height = input_shape[1] * factor
        new_width = input_shape[2] * factor
        new_shape = [input_shape[0], new_height, new_width, num_classes]
        output_shape = tf.stack(new_shape)

        # Get kernel size
        ksize = 2 * factor - factor % 2

        # Defines weights
        weights = tf.Variable(tf.truncated_normal(
            shape=[ksize, ksize, num_classes, in_features],
            mean=0,
            stddev=0.1),
            name='weights_'+str(name))

        # Apply backward strided convolution
        back_conv = tf.nn.conv2d_transpose(
            input,
            weights,
            output_shape,
            strides=[1,factor,factor,1],
            padding='SAME',
            name=name)

        return back_conv

    def load_vgg_16(self, features, weights):
        # Build VGG with VGG16 pretrained weights
        vgg = VGG(vgg16_weights_file=weights)
        # Return VGG16 layers
        return vgg.vgg16(features)

    def fcn32_vgg_16(self, features, weights, num_classes):
        # Load VGG16
        nn_layers = self.load_vgg_16(features, weights)

        # Replace dense layers with 1x1 convolution layers
        pool5 = nn_layers['pool5']
        pool5_conv1 = self.conv_layer_1x1(pool5, 4096, 'pool5_conv1')
        nn_layers['pool5_conv1'] = pool5_conv1
        pool5_conv2 = self.conv_layer_1x1(pool5_conv1, 4096, 'pool5_conv2')
        nn_layers['pool5_conv2'] = pool5_conv2

        # Upsampled to output segmentation using backward stride convolutions
        logits = self.back_conv_layer(pool5_conv2, 32, num_classes)
        probabilities = tf.nn.softmax(logits, name='fcn32_logits_to_softmax')

        # Return logits, probabilities, and layers for FCN-32
        return logits, probabilities, nn_layers

    def fcn16_vgg_16(self, features, weights, num_classes):
        # Load FCN32
        _, _, nn_layers = self.fcn32_vgg_16(features, weights, num_classes)

        # Add 1x1 convolution on top of pool4 layer
        pool4 = nn_layers['pool4']
        pool4_conv1 = self.conv_layer_1x1(pool4, num_classes, 'pool4_conv1')
        nn_layers['pool4_conv1'] = pool4_conv1

        # Fuse previous output with predictions computed on top of:
        #  - last CNN layer in FCN32 (with a 2x upsampling)
        pool5_conv2_2x = self.back_conv_layer(nn_layers['pool5_conv2'], 2, num_classes)
        fuse1 = tf.add(pool4_conv1, pool5_conv2_2x, 'fuse1')

        # Upsampled to output segmentation using backward stride convolutions
        logits = self.back_conv_layer(fuse1, 16, num_classes)
        probabilities = tf.nn.softmax(logits, name='fcn16_logits_to_softmax')

        # Return logits, probabilities, and layers for FCN-16
        return logits, probabilities, nn_layers

    def fcn8_vgg_16(self, features, weights, num_classes):
        # Load FCN16
        _, _, nn_layers = self.fcn16_vgg_16(features, weights, num_classes)

        # Add 1x1 convolution on top of pool3 layer
        pool3 = nn_layers['pool3']
        pool3_conv1 = self.conv_layer_1x1(pool3, num_classes, 'pool3_conv1')
        nn_layers['pool3_conv1'] = pool3_conv1

        # Fuse previous output with predictions computed on top of:
        #  - last CNN layer in FCN32 (with a 4x upsampling)
        #  - last CNN layer in FCN16 (with a 2x upsampling)
        pool4_conv1_2x = self.back_conv_layer(nn_layers['pool4_conv1'], 2, num_classes)
        fuse1 = tf.add(pool3_conv1, pool4_conv1_2x, 'fuse1')
        pool5_conv2_4x = self.back_conv_layer(nn_layers['pool5_conv2'], 4, num_classes)
        fuse2 = tf.add(fuse1, pool5_conv2_4x, 'fuse2')

        # Upsampled to output segmentation using backward stride convolutions
        logits = self.back_conv_layer(fuse2, 8, num_classes)
        probabilities = tf.nn.softmax(logits, name='fcn8_logits_to_softmax')

        # Return logits, probabilities, and layers for FCN-8
        return logits, probabilities, nn_layers
