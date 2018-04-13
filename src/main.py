import os
import glob
import random
import scipy.misc
import numpy as np
import tensorflow as tf
from fcn import FCN

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./data/", "Path to dataset")
tf.flags.DEFINE_integer("image_height", "2710", "Height of images in dataset")
tf.flags.DEFINE_integer("image_width", "3384", "Width of images in dataset")
tf.flags.DEFINE_string("vgg16_weights", "./weights/vgg16.npy", "Path to VGG16 weights")
tf.flags.DEFINE_integer("num_classes", "7", "Number of classes to predict")
tf.flags.DEFINE_integer("epochs", "50", "Number of epochs for training")
tf.flags.DEFINE_integer("batch_size", "20", "Batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for SGD Optimizer")
tf.flags.DEFINE_float("momentum", "0.9", "Momentum for SGD Optimizer")

def main(_):
    return None

if __name__ == '__main__':
    tf.app.run()
