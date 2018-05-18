import os
import glob
import shutil
import tensorflow as tf

def create_path_dir(path, clean=False):
    """
    Delete directory tree from path and (re-)create it.
    :param path: Path to clean and (re-)create
    :return:
    """

    dir = os.path.dirname(path)
    dir_exists = os.path.exists(dir)

    # Clean path diretories if exist
    if clean and dir_exists:
        shutil.rmtree(dir)

    # Create path directories if does not exist
    if not dir_exists:
        os.makedirs(dir)

def get_inputs(images_path, gt_images_path, height, width, batch_size):
    """
    Get inputs from CVPR2018 training datasets.
    :param images_path: Path to the CVPR2018 images data.
    :param gt_images_path: Path to the CVPR2018 ground truth images data.
    :param height: Image height.
    :param width: Image width.
    :param batch_size: Number of images per batch.
    :return:
        - images: Images. 4D tensor of [batch_size, height, width, 3] size.
        - gt_images: Ground truth images. 4D tensor of [batch_size, height, width, 1] size.
    """

    # Make a queue of file names including all the images files in
    # the CVPR2018 dataset directories
    train_images = tf.convert_to_tensor(glob.glob(images_path), dtype=tf.string)
    train_gt_images = tf.convert_to_tensor(glob.glob(gt_images_path), dtype=tf.string)
    filename_queues = tf.train.slice_input_producer([train_images, train_gt_images], shuffle=True)

    # Read whole image and ground truth image files from the queues
    raw_image = tf.read_file(filename_queues[0])
    raw_gt_image = tf.read_file(filename_queues[1])

    # Decode the image and ground truth image raw content
    image = tf.image.decode_image(raw_image, channels=3)
    gt_image = tf.image.decode_image(raw_gt_image, channels=1)

    # Preprocess image and ground truth image
    image, label = preprocess(image, gt_image, height, width)

    # Generate training batches
    with tf.name_scope('batch'):
        return generate_batches(image, label, batch_size, shuffle=True)

def preprocess(image, gt_image, height, width):
    """
    Perform preprocessing for image and ground truth image before feeding into network.
    :param image: Image 3D Tensor of shape [height, width, 3]
    :param annotation: Ground truth image 3D Tensor of shape [height, width, 1]
    :param height: Image height.
    :param width: Image width.
    :result:
        - image: Image. 3D tensor of [new_height, new_width, 3] size.
        - label: Label. 3D tensor of [new_height, new_width, num_classes] size.
    """

    # Convert the image dtypes to tf.float32 if needed
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Convert the image dtypes to tf.int32 if needed
    if gt_image.dtype != tf.int32:
        gt_image = tf.image.convert_image_dtype(gt_image, dtype=tf.int32)

    '''# Compute number of pixels needed to pad images
    # in order to respect FCN factor requirement
    top, bottom, left, right = get_paddings(height, width, 32)
    new_height = height + top + bottom
    new_width = width + left + right

    # Pad images if necessary
    image = tf.image.resize_image_with_crop_or_pad(image, new_height, new_width)
    gt_image = tf.image.resize_image_with_crop_or_pad(gt_image, new_height, new_width)
    '''

    # Subtract off the mean and divide by the variance of the pixels
    image = tf.image.per_image_standardization(image)

    # Shape TF tensors
    image.set_shape(shape=(height, width, 3))
    gt_image.set_shape(shape=(height, width, 1))

    # Dowscale images to save memory and time ;)
    image = tf.image.resize_images(image, size=(256, 256))
    gt_image = tf.squeeze(tf.image.resize_images(gt_image, size=(256, 256)))

    # Perform one-hot-encoding on the ground truth image
    label_ohe = one_hot_encode(gt_image)

    return image, label_ohe

def get_paddings(height, width, factor):
    """
    Compute number of pixels to add on each side to respect factor requirement.
    :param height: Image height
    :param width: Image width
    :param factor: Image dimensions must be divisible by factor
    :return: Number of pixels to add on each side
    """

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

    return (top, bottom, left, right)

def one_hot_encode(gt_image):
    """
    Generate label TF Tensor from ground truth image.
    :param gt_image: TF Tensor of the ground truth image of shape [height, width]
    :return: TF Tensor of the corresponding one hot label of shape [height, widht, num_classes+1]
    """

    # One hot encoding of each pixel according to the CVPR2018 classes
    label_ohe = list(map(lambda x: tf.to_float(tf.equal(gt_image // 1000, x)), cvpr2018_labels()))
    # Stack everything together
    return tf.stack(label_ohe, axis=-1)

def cvpr2018_labels():
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

def lyft_labels():
    """
    Build dictionnary with labels id/name to predict.
    :return: Dictionnary with label id - label name
    """

    return {
        0: 'None',
        7: 'Roads',
        10: 'Vehicles'
    }

def generate_batches(image, label, batch_size, shuffle):
    """
    Construct a queued batch of images and labels.
    :param image: 3-D Tensor of [height, width, 3] of type.float32.
    :param label: 3-D Tensor of [height, width, num_classes+1] of type.int32.
    :param batch_size: Number of images per batch.
    :param shuffle: boolean indicating whether to use a shuffling queue.
    :return:
        - images: Images. 4D tensor of [batch_size, height, width, 3] size.
        - labels: Labels. 4D tensor of [batch_size, height, width, num_classes+1] size.
    """

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            capacity=100,
            min_after_dequeue=50,
            allow_smaller_final_batch=True)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            allow_smaller_final_batch=True)

    # Display the training images in Tensorboard
    tf.summary.image('images', images)

    return images, labels
