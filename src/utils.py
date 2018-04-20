import os
import cv2
import glob
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def assure_path_exists(path):
    """
    Assure folders from path exist, or create them.
    :param path: Path to verify or create
    :return:
    """

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

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

def generate_batches(img_data, gt_img_data, data_ratio, batch_size):
    """
    Generate function to create batches of training data.
    :param img_data: Path to training images
    :param gt_img_data: Path to ground_truth images
    :param data_ratio: Ratio of training data to use
    :param batch_size: Batch size for training
    """

    # Enumerate every images/labels paths
    image_paths = glob.glob(img_data)
    label_paths = glob.glob(gt_img_data)
    labels = {get_filename_from_path(path, '_instanceIds'): path for path in label_paths}

    # Shuffle dataset
    random.shuffle(image_paths)

    # Only use a certain ratio of the dataset
    image_paths[:int(len(image_paths)*data_ratio)-1]

    # Generate batches
    for batch_i in range(0, len(image_paths), batch_size):
        images = []
        gt_images = []
        # For each image in current batch
        for image_path in image_paths[batch_i:batch_i + batch_size]:
            # Load image and pad it if necessary
            image, _ = load_image(image_path)
            gt_image, _ = load_image(labels[get_filename_from_path(image_path)], ground_truth=True)
            # Append to current batch array
            images.append(image)
            gt_images.append(gt_image)

        yield np.array(images), np.array(gt_images)

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

def get_labels_from_gt_images(annotations):
    """
    Generate labels TF Tensor from ground truth images.
    :param annotations: TF Tensor of the ground truth images of shape [batch, height, width]
    :return: TF Tensor of the corresponding one hot labels of shape [batch, height, widht, num_classes+1]
    """

    # One hot encoding of each pixel according to the CVPR2018 classes
    labels = list(map(lambda x: tf.to_float(tf.equal(annotations // 1000, x)), cvpr2018_lut()))
    # Add the instance ID
    labels.append(tf.to_float(annotations % 1000))
    # Stack everything together
    return tf.stack(labels, axis=3)
