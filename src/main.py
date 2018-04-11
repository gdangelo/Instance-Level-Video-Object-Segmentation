import time
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from vgg import VGG

# Read images
img1 = imread("./examples/catamaran.jpg").astype(np.float32)
img1 = img1 - np.mean(img1)
img2 = imread("./examples/electric_guitar.jpg").astype(np.float32)
img2 = img2 - np.mean(img2)

# Set dataset
x = tf.placeholder(tf.float32, shape=(None, 1080, 1920, 3))
resized = tf.image.resize_images(x, (224, 224))

# Get weights for VGG
vgg16_weights_file = './weights/vgg16.npy'
vgg19_weights_file = './weights/vgg19.npy'

# Load VGG models with pretrained weights
vgg = VGG(vgg16_weights_file, vgg19_weights_file)
vgg_16_logits = vgg.vgg16(resized)
vgg_16_top_k = tf.nn.top_k(vgg_16_logits, k=5)
vgg_19_logits = vgg.vgg19(resized)
vgg_19_top_k = tf.nn.top_k(vgg_19_logits, k=5)

# Test models
with tf.Session() as sess:
    t1 = time.time()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Fire!
    vgg16_output = sess.run(vgg_16_top_k, feed_dict={ x: [img1, img2] })
    vgg19_output = sess.run(vgg_19_top_k, feed_dict={ x: [img1, img2] })

    t2 = time.time()

    print('Top 5 probs for VGG16: ')
    print(vgg16_output)
    print('Top 5 probs for VGG19: ')
    print(vgg19_output)

    print('Models run in {0:.2f}s'.format(t2-t1))
