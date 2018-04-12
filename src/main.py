import time
import numpy as np
import tensorflow as tf
from scipy.misc import imread
import matplotlib.pyplot as plt
from fcn import FCN

# Read images
img1 = imread("./examples/catamaran.jpg").astype(np.float32)
img1 = img1 - np.mean(img1)
img2 = imread("./examples/electric_guitar.jpg").astype(np.float32)
img2 = img2 - np.mean(img2)

# Set dataset
x = tf.placeholder(tf.float32, shape=(None, 1080, 1920, 3))
resized = tf.image.resize_images(x, (224, 224))

# Load FCN models with pretrained weights for VGG16
fcn_vgg_16 = FCN().fcn_vgg_16('./weights/vgg16.npy', resized)

# Test models
with tf.Session() as sess:
    t1 = time.time()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Fire!
    output = sess.run(fcn_vgg_16, feed_dict={ x: [img1, img2] })
    t2 = time.time() - t1
    print('Model run in {0:.2f}s'.format(t2))

    # Visualize
    for i in range(output.shape[0]):
        mat = plt.matshow(output[i])
        plt.show()
