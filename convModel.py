from numpy import *
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)
    
def conv2d(x,w):
    return tf.nn.conv2d(x,w,[1,1,1,1],padding = 'same')#batch height width channel
    
def max_pool(x,klen):
    return tf.nn.max_pool(x,ksize = [1,klen,klen,1], strides = [1,klen,klen,1])
    
def covNetwork(train_x,train_y):