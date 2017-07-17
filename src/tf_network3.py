import numpy as np
import tensorflow as tf
from tf.nn import softmax
from tf.nn import sigmoid

class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tf.placeholder(tf.float32, [None, layers[0]]) # None since that dim can be of any length
        self.y = tf.placeholder(tf.float32, [None, layers[-1]])
        # construct the computation graph up to the network output
        inpt = self.x
        for layer in self.layers:
            layer.set_inpt(inpt, mini_batch_size)
            inpt = layer.output
        self.output = self.layers[-1].output


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid):
        self.n_in
        self.n_out
        self.w = tf.Variable(tf.random_normal(n_in, n_out))
        self.b = tf.Variable(tf.random_normal(n_out))
        self.params = [self.w, self.b]

    # construct the computation graph up to the output
    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = activation_fn(tf.matmul(self.inpt, self.w) + self.b)

    def cost(self, net):
        #todo
        pass
