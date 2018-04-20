import os
import cv2
import math
import glob
import random
import scipy.misc
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.misc import imread
from fcn import FCN

script_dir = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", os.path.join(script_dir, "../data/"), "Path to dataset")
tf.flags.DEFINE_string("vgg16_weights", os.path.join(script_dir, "../weights/vgg16.npy"), "Path to VGG16 weights")
tf.flags.DEFINE_string("vgg19_weights", os.path.join(script_dir, "../weights/vgg19.npy"), "Path to VGG19 weights")
tf.flags.DEFINE_float("data_ratio", "1", "Ratio of training data to use")
tf.flags.DEFINE_integer("image_height", "2710", "Height of images in dataset")
tf.flags.DEFINE_integer("image_width", "3384", "Width of images in dataset")
tf.flags.DEFINE_integer("num_classes", "8", "Number of classes to predict")
tf.flags.DEFINE_integer("epochs", "50", "Number of epochs for training")
tf.flags.DEFINE_integer("batch_size", "1", "Batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for SGD Optimizer")
tf.flags.DEFINE_float("momentum", "0.9", "Momentum for SGD Optimizer")

def assure_path_exists(path):
    """
    Assure folders from path exist, or create them.
    :param path: Path to verify or create
    :return:
    """

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def cvpr2018_lut():
    """
    Build dictionnary with labels id/name to predict.
    :return: Dictionnary with label id - label name
    """

    return {
        0: 'others',
        33: 'car',
        34: 'motorcycle',
        35: 'bicycle',
        36: 'pedestrian',
        38: 'truck',
        39: 'bus',
        40: 'tricycle'
    }

def get_filename_from_path(path, str_to_remove=None):
    """
    Retrieve filename from path without extension.
    :param path: Path to file
    :param str_to_remove: String to remove from filename
    :return: Filename without extension
    """

    # Get filename without extension
    filename_w_ext = os.path.basename(path)
    filename, file_extension = os.path.splitext(filename_w_ext)

    # Remove occurences in the filename
    if str_to_remove is not None:
        filename = filename.replace(str_to_remove, '')

    return filename

def load_image(path, ground_truth=False, pad=True, factor=32):
    """
    Load image from path and pad it on each side to respect factor requirement.
    :param path: Path to the image to read
    :param ground_truth: If set to True, image is read as grayscale
    :param pad: If set to True, image is padded to respect factor
    :param factor: Image dimensions must be divisible by factor
    """

    # Read image from path
    if ground_truth is True:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Return image if no padding is required
    if pad is not True:
        return img

    # Compute pixels needed to pad image
    height, width = img.shape[:2]

    if height % factor == 0:
        top = 0
        bottom = 0
    else:
        pixels = factor - height % factor
        top = int(pixels / 2)
        bottom = pixels - top

    if width % factor == 0:
        left = 0
        right = 0
    else:
        pixels = factor - width % factor
        left = int(pixels / 2)
        right = pixels - left

    # Draw border around image
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    # Return padded image and padding values
    return img, (top, bottom, left, right)

def crop_image(img, pads):
    """
    Crop image according to padding values on each side.
    :param img: Image to crop on each side
    :param pads: Tuple of (top, bottom, left, right) padding values
    :return: Cropped image
    """

    # Get image dimensions and padding values
    top, bottom, left, right = pads
    height, width = img.shape[:2]
    # Crop image
    return img[top:height-bottom, left:width-right]

def generate_batches():
    """
    Generate function to create batches of training data.
    """

    # Enumerate every images/labels paths
    image_paths = glob.glob(os.path.join(FLAGS.data_dir, 'train_color', '*.jpg'))
    label_paths = glob.glob(os.path.join(FLAGS.data_dir, 'train_label', '*_instanceIds.png'))
    labels = {get_filename_from_path(path, '_instanceIds'): path for path in label_paths}

    # Shuffle dataset
    random.shuffle(image_paths)

    # Only use a certain ratio of the dataset
    image_paths[:int(len(image_paths)*FLAGS.data_ratio)-1]

    # Generate batches
    for batch_i in range(0, len(image_paths), FLAGS.batch_size):
        images = []
        gt_images = []
        # For each image in current batch
        for image_path in image_paths[batch_i:batch_i + FLAGS.batch_size]:
            # Load image and pad it if necessary
            image = load_image(image_path)
            gt_image = load_image(labels[get_filename_from_path(image_path)], ground_truth=True)
            gt_image = gt_image // 1000
            # Append to current batch array
            images.append(image)
            gt_images.append(gt_image)

        yield np.array(images), np.array(gt_images)

def optimize(logits, labels):
    """
    Build the TensorFLow loss and optimizer operations.
    :param logits: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label image
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshape labels from [height, width] to [height, widht, num_classes]
    labels_2d = list(map(lambda x: tf.equal(labels, x), cvpr2018_lut()))
    labels_2d_stacked = tf.stack(labels_2d, axis=3)
    labels_2d_stacked_float = tf.to_float(labels_2d_stacked)

    # Compute the loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_2d_stacked_float)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # Optimize the loss
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    train_op = optimizer.minimize(cross_entropy_loss)

    return train_op, cross_entropy_loss

def train_nn(sess, train_op, cross_entropy_loss):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    """

    # Get number of batches for visualization purpose
    n_samples = len(glob.glob(os.path.join(FLAGS.data_dir, 'train_color', '*.jpg')))
    n_batches = int(math.ceil(float(n_samples) / FLAGS.batch_size))

    for i in range(FLAGS.epochs):
        # Generate batches
        batches = tqdm(generate_batches(),
                        desc='Epoch {}/{} (loss _.___)'.format(i+1, FLAGS.epochs),
                        total=n_batches)
        # Run the training pipeline for each batch
        for x_batch, y_batch in batches:
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={ features: x_batch, labels: y_batch })
            # Print loss after each batch has been trained
            batches.set_description('Epoch {}/{} (loss {:.3f})'.format(i+1, FLAGS.epochs, loss))

def main(_):
    # Define TF placeholders for training images and labels
    features = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 3])
    labels = tf.placeholder(tf.int32, shape=[None, FLAGS.image_height, FLAGS.image_width])

    # Load FCN model with pretrained weights for VGG16
    logits, _ = FCN().fcn32_vgg_16(features, FLAGS.vgg16_weights, FLAGS.num_classes)
    #logits, _ = FCN().fcn16_vgg_16(features, FLAGS.vgg16_weights, FLAGS.num_classes)
    #logits, _ = FCN().fcn8_vgg_16(features, FLAGS.vgg16_weights, FLAGS.num_classes)

    # Build the TF loss and optimizer
    train_op, cross_entropy_loss = optimize(logits, labels)

    # Train FCN
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        train_nn(sess, train_op, cross_entropy_loss)

        # Save model
        path = './run/fcn32/fcn32_vgg_16.ckpt'
        assure_path_exists(path)
        saver.save(sess, path)
        print("Model saved")

if __name__ == '__main__':
    tf.app.run()
