import os
import glob
import math

from datetime import datetime
import time

import utils
import tensorflow as tf
from models.fcn import FCN

script_dir = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.flags.FLAGS

# Data directories and paths
tf.flags.DEFINE_string('train_dir', os.path.join(script_dir, '../cvpr2018_train/'), "Directory where to write event logs and checkpoint.")
tf.flags.DEFINE_string('data_dir', os.path.join(script_dir, '../data/'), "Path to the CVPR2018 data directory.")
tf.flags.DEFINE_string('images_path', os.path.join(FLAGS.data_dir, 'train_color', '*.jpg'), "Path to training images.")
tf.flags.DEFINE_string('gt_images_path', os.path.join(FLAGS.data_dir, 'train_label', '*_instanceIds.png'), "Path to label images.")

# Data dimensions
tf.flags.DEFINE_integer('image_height', 3384, "Height of images in dataset.")
tf.flags.DEFINE_integer('image_width', 2710, "Width of images in dataset.")

# Pre-trained VGG weights files
tf.flags.DEFINE_string('vgg16_weights', os.path.join(script_dir, '../weights/vgg16.npy'), "Path to VGG16 weights.")
tf.flags.DEFINE_string('vgg19_weights', os.path.join(script_dir, '../weights/vgg19.npy'), "Path to VGG19 weights.")

# Training hyperparameters
tf.flags.DEFINE_string('model', 'fcn8', "Model to use for training.")
tf.flags.DEFINE_integer('num_classes', 8, "Number of classes to predict.")
tf.flags.DEFINE_integer('batch_size', 16, "Number of images to process in a batch.")
tf.flags.DEFINE_float('initial_learning_rate', 1e-4, "Learning rate for SGD Optimizer.")
tf.flags.DEFINE_float('learning_rate_decay_factor', 1e-1, "Learning rate decay factor.")
tf.flags.DEFINE_float('num_epochs_per_decay', 50, "Epochs after which learning rate decays.")
tf.flags.DEFINE_integer('num_epochs', 250, "Number of epochs to run.")
tf.flags.DEFINE_float('momentum', 0.9, "Momentum for SGD Optimizer.")
tf.flags.DEFINE_float('beta', 0.01, "Beta for L2 regularization.")
tf.flags.DEFINE_boolean('clean_train_dir', False, "Clean training directory before training the model again.")

# Logging parameters
tf.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.flags.DEFINE_integer('log_frequency', 10, "How often to log results to the console.")

class _LoggerHook(tf.train.SessionRunHook):
    """
    TF Hook used in training session to log metrics and runtime.
    """

    def __init__(self, xentropy_loss, accuracy, mean_iou, lr, num_batches_per_epoch):
        self._xentropy_loss = xentropy_loss
        self._accuracy = accuracy
        self._mean_iou = mean_iou
        self._lr = lr
        self._num_batches_per_epoch = num_batches_per_epoch

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        # Ask for metrics values
        return tf.train.SessionRunArgs([self._xentropy_loss, self._accuracy,
                                        self._mean_iou, self._lr])

    def after_run(self, run_context, run_values):
        # Display info on every new batch
        if self._step % self._num_batches_per_epoch == 0:
            learning_rate_value = run_values.results[3]
            print('\n-- Epoch %s/%s, learning rate = %f' %
                  (int(self._step/self._num_batches_per_epoch) + 1,
                  FLAGS.num_epochs,
                  learning_rate_value))

        # Log training summary every 'log_frequency' steps
        if self._step % min(self._num_batches_per_epoch, FLAGS.log_frequency) == 0:
            # Compute how long the training steps lasted
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            # Retrieve metrics values
            loss_value = run_values.results[0]
            accuracy_value = run_values.results[1]
            mean_iou_value = run_values.results[2]

            # Compute how the training performed
            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = float(duration / FLAGS.log_frequency)

            # Display results
            format_str = ('%s: step %d, loss = %.2f, accuracy = %.2f, IOU = %.2f '
                          '(%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (datetime.now(), self._step, loss_value,
                                accuracy_value, mean_iou_value,
                                examples_per_sec, sec_per_batch))

def train():
    """
    Train neural network for a number of steps.
    """

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CVPR2018.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down
        with tf.device('/cpu:0'):
            images, labels_ohe = utils.get_inputs(FLAGS.images_path, FLAGS.gt_images_path,
                                      FLAGS.image_height, FLAGS.image_width,
                                      FLAGS.batch_size)

        # Load FCN model with pre-trained weights for VGG16
        logits, probabilities = load_fcn(images)

        # Calculate loss
        xentropy_loss = loss(logits, labels_ohe)

        # Define learning rate
        num_batches_per_epoch = math.ceil(len(glob.glob(FLAGS.images_path)) / FLAGS.batch_size)
        lr = learning_rate(num_batches_per_epoch, global_step)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters
        train_op = optimize(xentropy_loss, lr, global_step)

        # Define metrics to predict
        predictions = tf.argmax(probabilities, -1)
        labels = tf.argmax(labels_ohe, -1)
        accuracy, accuracy_update = tf.metrics.accuracy(labels, predictions)
        mean_iou, mean_iou_update = tf.metrics.mean_iou(labels, predictions, FLAGS.num_classes)
        metrics_op = tf.group(accuracy_update, mean_iou_update)

        # Define config and hooks for the training session
        max_steps = int(FLAGS.num_epochs * num_batches_per_epoch)

        config = tf.ConfigProto()
        config.log_device_placement = FLAGS.log_device_placement
        config.gpu_options.allow_growth = True

        hooks =[tf.train.StopAtStepHook(last_step=max_steps),
                tf.train.NanTensorHook(xentropy_loss),
                _LoggerHook(xentropy_loss, accuracy, mean_iou, lr, num_batches_per_epoch)]

        # Run the training session
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=hooks,
            config=config) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run([train_op, accuracy, mean_iou, metrics_op, lr])

def load_fcn(images):
    """
    Load FCN model with pre-trained VGG16 weigths.
    :return: Logits and probabilities from FCN output
    """

    if FLAGS.model == 'fcn32':
        print("Building FCN32_VGG16 model...")
        logits, probabilities, _ = FCN().fcn32_vgg_16(images, FLAGS.vgg16_weights, FLAGS.num_classes)
    elif FLAGS.model == 'fcn16':
        print("Building FCN16_VGG16 model...")
        logits, probabilities, _ = FCN().fcn16_vgg_16(images, FLAGS.vgg16_weights, FLAGS.num_classes)
    elif FLAGS.model == 'fcn8':
        print("Building FCN8_VGG16 model...")
        logits, probabilities, _ = FCN().fcn8_vgg_16(images, FLAGS.vgg16_weights, FLAGS.num_classes)
    else:
        raise ValueError('Unknown model: ' + FLAGS.model)

    return logits, probabilities

def learning_rate(num_batches_per_epoch, global_step):
    """
    Define exponentially decaying learning rate.
    :param global_step: Variable counting the number of training steps processed
    :return: Learning rate
    """

    # Compute decay steps for learning rate
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Define exponentially decaying learning rate
    return tf.train.exponential_decay(
        learning_rate = FLAGS.initial_learning_rate,
        global_step = global_step,
        decay_steps = decay_steps,
        decay_rate = FLAGS.learning_rate_decay_factor,
        staircase = True)

def loss(logits, labels_ohe):
    """
    Build the TensorFLow loss operations.
    :param logits: TF tensor of the last layer in the neural network
    :param labels_ohe: TF tensor of the labels one hot encoded
    :return: Loss TF tensor
    """

    # Compute the cross entropy loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_ohe)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # Add L2 regularization
    regularizers = [tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'weights' in var.name]
    return tf.reduce_mean(cross_entropy_loss + FLAGS.beta * tf.add_n(regularizers))


def optimize(total_loss, lr, global_step):
    """
    Build the TensorFLow optimizer operation.
    :param total_loss: Total loss
    :param global_step: Variable counting the number of training steps processed
    :param num_batches_per_epoch: Number of batches per epoch
    :return: Operation for training
    """

    # Optimize the loss
    optimizer = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
    return optimizer.minimize(total_loss)

def main(argv=None):
    """Run main function."""

    # Clean and create training directory
    utils.create_path_dir(FLAGS.train_dir, FLAGS.clean_train_dir)
    # Do training
    train()

if __name__ == "__main__":
    tf.app.run()
