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

def get_filename_from_path(path, str_to_remove=None):
    # Get filename without extension
    filename_w_ext = os.path.basename(path)
    filename, file_extension = os.path.splitext(filename_w_ext)

    # Remove occurences in the filename
    if str_to_remove is not None:
        filename = filename.replace(str_to_remove, '')

    return filename

def generate_batches():
    # Enumerate every images/labels paths
    image_paths = glob.glob(os.path.join(FLAGS.data_dir, 'train_color', '*.jpg'))
    label_paths = glob.glob(os.path.join(FLAGS.data_dir, 'train_label', '*_instanceIds.png'))
    labels = {get_filename_from_path(path, '_instanceIds'): path for path in label_paths}

    # Shuffle dataset
    random.shuffle(image_paths)

    # Generate batches
    for batch_i in range(0, len(image_paths), FLAGS.batch_size):
        images = []
        gt_images = []
        # Read images and append to current batch
        for image_path in image_paths[batch_i:batch_i + FLAGS.batch_size]:
            image = scipy.misc.imread(image_path)
            gt_image = scipy.misc.imread(labels[get_filename_from_path(image_path)])
            images.append(image)
            gt_images.append(gt_image)

        yield np.array(images), np.array(gt_images)

def main(_):
    return None

if __name__ == '__main__':
    tf.app.run()
