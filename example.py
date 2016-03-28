import tensorflow as tf
from spatial_transformer import transformer
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from tf_utils import conv2d, linear, weight_variable, bias_variable

# Preprocessing
# Create a batch of three images (1600 x 1200)
im = ndimage.imread('./data/cat.jpg')
im = im / 255.
im = im.reshape(1, 1200, 1600, 3)
im = im.astype('float32')
# Simulate batch
batch = np.append(im, im, axis=0)
batch = np.append(batch, im, axis=0)

num_batch = 3
x = tf.placeholder(tf.float32, [None, 1200, 1600, 3])
x = tf.cast(batch,'float32')

num_batch = 3
x = tf.placeholder(tf.float32, [None, 1200, 1600, 3])
x = tf.cast(batch,'float32')

# Create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):

    # %% Create a fully-connected layer:
    n_fc = 6 
    W_fc1 = tf.Variable(tf.zeros([1200 * 1600 * 3, n_fc]), name='W_fc1')
    initial = np.array([[0.5,0, 0],[0,0.5,0]]) 
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
    x_flatten = tf.reshape(x,[-1,1200 * 1600 * 3])
    #h_fc1 = tf.nn.relu(tf.matmul(x_flatten, W_fc1) + b_fc1)
    h_fc1 = tf.matmul(tf.zeros([num_batch ,1200 * 1600 * 3]), W_fc1) + b_fc1
    h_trans = transformer(x, h_fc1, downsample_factor=2)

# Run session
sess = tf.Session()
sess.run(tf.initialize_all_variables())
y = sess.run(h_trans, feed_dict={x: batch})

plt.imshow(y[0])