import tensorflow as tf
from spatial_transformer import transformer
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def conv2d(x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           activation=lambda x: x,
           bias=True,
           padding='SAME',
           name="Conv2D"):
    """2D Convolution with options for kernel size, stride, and init deviation.
    Parameters
    ----------
    x : Tensor
        Input tensor to convolve.
    n_filters : int
        Number of filters to apply.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID'
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Convolved input.
    """
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k_h, k_w, x.get_shape()[-1], n_filters],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            b = tf.get_variable(
                'b', [n_filters],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = conv + b
        return conv
    
def linear(x, n_units, scope=None, stddev=0.02,
           activation=lambda x: x):
    """Fully-connected network.
    Parameters
    ----------
    x : Tensor
        Input tensor to the network.
    n_units : int
        Number of units to connect to.
    scope : str, optional
        Variable scope to use.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    Returns
    -------
    x : Tensor
        Fully-connected output.
    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x, matrix))
    
# %%
def weight_variable(shape):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    #initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    initial = tf.zeros(shape)
    return tf.Variable(initial)

# %%
def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

# Preprocessing
# Create a batch of three images (1600 x 1200)
im = ndimage.imread('cat.jpg')
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
#     filter_size = 3
#     n_filters_1 = 3
#     W_conv1 = weight_variable([filter_size, filter_size, 3, n_filters_1])

#     # %% Bias is [output_channels]
#     b_conv1 = bias_variable([n_filters_1])

#     # %% Now we can build a graph which does the first layer of convolution:
#     # we define our stride as batch x height x width x channels
#     # instead of pooling, we use strides of 2 and more layers
#     # with smaller filters.
#     h_conv1 = tf.nn.relu(
#         tf.nn.conv2d(input=x,
#                      filter=W_conv1,
#                      strides=[1, 1, 1, 1],
#                      padding='SAME') +
#         b_conv1)
#     h_conv1_trans = tf.transpose(h_conv1, perm=[0, 3, 1, 2])
#     # %% We'll now reshape so we can connect to a fully-connected layer:
#     h_conv2_flat = tf.reshape(h_conv1, [-1, 1200 * 1600 * 3])

    # %% Create a fully-connected layer:
    n_fc = 6 
    W_fc1 = tf.Variable(tf.zeros([1200 * 1600 * 3, n_fc]), name='W_fc1')
    initial = np.array([[0.5,0, 0],[0,0.5,0]]) 
    initial = initial.flatten()
    b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
    b_fc1 = tf.cast(b_fc1, 'float32') # cast it to float32
    x_flatten = tf.reshape(x,[-1,1200 * 1600 * 3])
    #h_fc1 = tf.nn.relu(tf.matmul(x_flatten, W_fc1) + b_fc1)
    h_fc1 = tf.matmul(tf.zeros([num_batch ,1200 * 1600 * 3]), W_fc1) + b_fc1
    h_trans = transformer(x, h_fc1, downsample_factor=2)

# Run session
sess = tf.Session()
sess.run(tf.initialize_all_variables())
y = sess.run(h_trans, feed_dict={x: batch})

plt.imshow(y[0])