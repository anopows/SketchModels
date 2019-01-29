import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.slim import conv2d
from tensorflow.contrib.layers import xavier_initializer as xav_init

    
def normalize(images, axis=[2]):
    """ Modified from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/image_ops_impl.py
    """
    num_pixels = math_ops.reduce_prod(array_ops.shape(images)[axis[0]]) # TODO multiple axis

    images = math_ops.cast(images, dtype=tf.float32)
    # E[X]
    image_mean = math_ops.reduce_mean(images, axis=axis, keepdims=True) # [?,?,1]

    # X^2
    squares = math_ops.square(images) # [?,?,pixels] 
    # E[X^2] - E[X]^2
    variance = (math_ops.reduce_mean(squares, axis=axis, keepdims=True) -
                math_ops.square(image_mean))
    # guarantee non-null (?)
    variance = gen_nn_ops.relu(variance)
    stddev = math_ops.sqrt(variance) # [?,?,1]

    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, tf.float32)) # [1]
    min_stddev = tf.fill(tf.shape(stddev), min_stddev) # [?,?,1]

    # adjust std dev, mean
    pixel_value_scale = math_ops.maximum(stddev, min_stddev)
    pixel_value_offset = image_mean

    # apply
    images = math_ops.subtract(images, pixel_value_offset)
    images = math_ops.div(images, pixel_value_scale)
    return images

# Helper fns for threesplit way of feeding
def _distance_squared(lts1, lts2): # Works for both simple vectors, or a batch of vectors
    difference = lts1 - lts2
    difference_sqrs = tf.square(difference)
    return tf.reduce_sum(difference_sqrs, -1)


def _fcn(logits, logits_out, name, activation=tf.nn.relu,
         training=True, trainable=True, bn_before=False, bn_after=False):
    if bn_before: logits = tf.layers.batch_normalization(logits,  
                                                         training=training, trainable=trainable,
                                                         name=name + '_bn_before')
    logits = tf.layers.dense(logits, logits_out, activation=activation, name=name)
    if bn_after:  logits = tf.layers.batch_normalization(logits,
                                                         training=training, trainable=trainable,
                                                         name=name + '_bn_after')
    return logits

def classification(logits, layer_sizes=[64,10], name='classifier',
                   bnorm_before=False, bnorm_middle=False, training=True, trainable=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i,layer_size in enumerate(layer_sizes):
            layer_name = 'layer' + str(i+1)
            
            if i==0 and bnorm_before:
                logits = _fcn(logits, layer_size, layer_name,
                              training=training, trainable=trainable, bn_before=True, bn_after=bnorm_middle)
            elif (i+1) == len(layer_sizes): # last layer
                return _fcn(logits, layer_size, layer_name, activation=None,
                            trainable=trainable, training=training) # No BatchNormalization
            else:
                logits = _fcn(logits, layer_size, layer_name,
                              training=training, trainable=trainable, bn_after=bnorm_middle)
        
