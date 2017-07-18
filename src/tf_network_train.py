import sys
import argparse
import tensorflow as tf
from tf_network3 import Network
from tf_network3 import FullyConnectedLayer
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):

    # fetch the training data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    training_data, validation_data, test_data = mnist.train, mnist.validation, mnist.test

    # hyperparameter
    epochs = 1
    mini_batch_size = 10
    eta = 0.5

    layers = [FullyConnectedLayer(784, 30), FullyConnectedLayer(30, 10)]
    net = Network(layers, mini_batch_size)
    net.SGD(training_data, epochs, mini_batch_size, eta, validation_data, test_data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
