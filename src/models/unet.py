import numpy as np
import tensorflow as tf
from models.vgg import VGG

class UNET:
    """
    UNET class
    """

    def load_vgg_16(self, features, weights):
        # Build VGG with VGG16 pretrained weights
        vgg = VGG(vgg16_weights_file=weights)
        # Return VGG16 layers
        return vgg.vgg16(features)

    def conv_layer(self, input, n_filters, ksize=(3,3), strides=(1,1), padding='VALID', name='conv'):
        with tf.name_scope(name):
            input_shape = input.get_shape().as_list()

            # Define weights and bias
            weights = tf.Variable(tf.truncated_normal(
                shape=[*ksize, input_shape[3], n_filters],
                mean=0,
                stddev=0.1),
                'weights_'+str(name))
            bias = tf.Variable(tf.zeros(n_filters), name='bias_'+str(name))

            # Apply convolution layer
            conv = tf.nn.conv2d(input, weights, strides=[1, *strides, 1], padding=padding, name=name)
            # Add bias
            conv = tf.nn.bias_add(conv, bias)
            # Apply activation function
            return tf.nn.relu(conv)

    def max_pool_layer(self, input, ksize=(2,2), strides=(2,2), padding='SAME', name='pool'):
        with tf.name_scope(name):
            return tf.nn.max_pool(input, ksize=[1, *ksize, 1], strides=[1, *strides, 1], padding=padding, name=name)

    def conv_block(self, input, n_filters, pooling=True, name=None):
        block = input
        # Apply two 3x3 convolutions (unpadded)
        for i, F in enumerate(n_filters):
            block = self.conv_layer(block, F, name='{}_conv_{}'.format(name, i+1))

        # Return block if no pooling is required
        if pooling is False:
            return block

        # Apply 2x2 max pooling with stride 2
        pool = self.max_pool_layer(block, name='{}_pool'.format(name))

        return block, pool

    def back_conv_layer(self, input, factor, num_classes, name='back_conv'):
        with tf.name_scope(name):
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

    def up_block(self, bottom_input, copy_input, name=None):
        copy_size = copy_input.get_shape().as_list()
        copy_height = copy_size[1]
        copy_width = copy_size[2]
        n_filters = copy_size[3]

        # Apply upsampling by a 2x2 back convolution
        up_conv = self.back_conv_layer(bottom_input, 2, n_filters, name='{}_back_conv'.format(name))

        # Crop and copy
        target_size = tf.shape(up_conv)
        target_height = target_size[1]
        target_width = target_size[2]
        offset_height = tf.to_int32((copy_height - target_height) / 2)
        offset_width = tf.to_int32((copy_width - target_width) / 2)
        cropped = tf.image.crop_to_bounding_box(copy_input, offset_height, offset_width, target_height, target_width)
        return tf.concat([up_conv, cropped], axis=-1, name='{}_concat'.format(name))

    def unet(self, features, weights, num_classes):

        # --- Contracting path: VGG16 (pre-trained) ---
        vgg16_layers = self.load_vgg_16(features, weights)

        # --- Expansive path ---
        up_block_1 = self.up_block(vgg16_layers['conv5_3'], vgg16_layers['conv4_3'], name='up_block_1')
        block_6 = self.conv_block(up_block_1, [512, 512], pooling=False, name='block_6')

        up_block_2 = self.up_block(block_6, vgg16_layers['conv3_3'], name='up_block_1')
        block_7 = self.conv_block(up_block_2, [256, 256], pooling=False, name='block_7')

        up_block_3 = self.up_block(block_7, vgg16_layers['conv2_2'], name='up_block_1')
        block_8 = self.conv_block(up_block_3, [128, 128], pooling=False, name='block_8')

        up_block_4 = self.up_block(block_8, vgg16_layers['conv1_2'], name='up_block_1')
        block_9 = self.conv_block(up_block_4, [64, 64], pooling=False, name='block_9')

        logits = self.conv_layer(block_9, num_classes, ksize=(1,1), name='unet')
        probabilities = tf.nn.softmax(logits, name='unet_logits_to_softmax')

        # Return logits and probabilities for U-Net
        return logits, probabilities
