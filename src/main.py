import os
import glob
import math
import utils
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from fcn import FCN

script_dir = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", os.path.join(script_dir, "../data/"), "Path to dataset")
tf.flags.DEFINE_string("img_data", os.path.join(FLAGS.data_dir, 'train_color', '*.jpg'), "Path to training images")
tf.flags.DEFINE_string("gt_img_data", os.path.join(FLAGS.data_dir, 'train_label', '*_instanceIds.png'), "Path to ground_truth images")
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

def optimize(logits, annotations):
    """
    Build the TensorFLow loss and optimizer operations.
    :param logits: TF Tensor of the last layer in the neural network
    :param annotations: TF Placeholder for the correct annotations image
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshape labels from [height, width] to [height, widht, num_classes+1]
    labels = utils.get_labels_from_gt_images(annotations)

    # Compute the loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
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
    n_samples = len(glob.glob(FLAGS.img_data))
    n_batches = int(math.ceil(float(n_samples) / FLAGS.batch_size))

    for i in range(FLAGS.epochs):
        # Generate batches
        batches = tqdm(utils.generate_batches(FLAGS.img_data, FLAGS.gt_img_data, FLAGS.data_ratio, FLAGS.batch_size),
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

        #train_nn(sess, train_op, cross_entropy_loss)

        # Save model
        path = './run/fcn32/fcn32_vgg_16.ckpt'
        utils.assure_path_exists(path)
        saver.save(sess, path)
        print("Model saved")

if __name__ == '__main__':
    tf.app.run()
