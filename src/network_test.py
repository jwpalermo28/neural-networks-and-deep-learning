import mnist_loader
import matrix_based_network
import network
import numpy as np
import time

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

my_net = matrix_based_network.Network([784, 30, 10])
net = network.Network([784, 30, 10])

start_time = time.time()
my_net.SGD(training_data, 1, 10, 0.5, test_data=test_data)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
net.SGD(training_data, 1, 10, 0.5, test_data=test_data)
print("--- %s seconds ---" % (time.time() - start_time))
