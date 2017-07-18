import numpy as np
import tensorflow as tf

class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tf.placeholder(tf.float32, [None, layers[0].n_in]) # None since dim can be of any length
        self.y = tf.placeholder(tf.float32, [None, layers[-1].n_out])
        # construct the computation graph up to the network output
        inpt = self.x
        for layer in self.layers:
            layer.set_inpt(inpt, mini_batch_size)
            inpt = layer.output
        self.output = self.layers[-1].output

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):

        # compute number of minibatches per epoch
        num_training_batches = training_data.images.shape[0] // mini_batch_size
        num_validation_batches = validation_data.images.shape[0] // mini_batch_size
        num_test_batches = test_data.images.shape[0] // mini_batch_size

        # define the cost function and optimization procedure
        cost = self.layers[-1].cost(self)
        train_step = tf.train.GradientDescentOptimizer(eta).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # train
        best_validation_accuracy = 0.0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(epochs):
                for mini_batch_i in range(num_training_batches):
                    iteration = num_training_batches * epoch_i + mini_batch_i
                    if iteration % 1000 == 0:
                        print("Training mini-batch number " + str(mini_batch_i))
                    batch_xs, batch_ys = training_data.next_batch(mini_batch_size)
                    train_step.run(feed_dict= {self.x: batch_xs, self.y: batch_ys})
                    # for a quick check that this implementation works
                    if iteration % 100 == 0:
                        batch_xs, batch_ys = validation_data.next_batch(mini_batch_size)
                        print sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys})
            #             validation_accuracy = np.mean(
            #                 [sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys})
            #                 for batch_xs, batch_ys in [validation_data.next_batch(mini_batch_size) for _ in range(num_validation_batches)])
            #             print("Epoch " + str(epoch_i) + ": validation accuracy " + str(validation_accuracy))
            #             if validation_accuracy >= best_validation_accuracy:
            #                 print("This is the best validation accuracy to date.")
            #                 best_validation_accuracy = validation_accuracy
            #                 best_iteration = iteration
            #                 if test_data:
            #                     test_accuracy = np.mean(
            #                         [sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys})
            #                         for batch_xs, batch_ys in test_data.next_batch(mini_batch_size)])
            #                     print('The corresponding test accuracy is {0:.2%}'.format(
            #                         test_accuracy))
            # print("Finished training network.")
            # print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            #     best_validation_accuracy, best_iteration))
            # print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=tf.nn.sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.w = tf.Variable(tf.random_normal([n_in, n_out]))
        self.b = tf.Variable(tf.random_normal([n_out]))
        self.params = [self.w, self.b]

    # construct the computation graph up to the output
    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = self.activation_fn(tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.argmax(self.output, axis=1)

    def cost(self, net):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.output - net.y), axis=1))
