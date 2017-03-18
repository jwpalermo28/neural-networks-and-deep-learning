import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU
import sys

# get the learning rate as a command line argument
eta = float(sys.argv[1])
save_model_as = '../training_results/' + sys.argv[2] + '/model.pkl'

# test network3 ----------------------------------------------------------------

# load data for network3
data_dir = '../../Kaggle/cats_and_dogs/data224/labelled_data/'
training_data, validation_data, test_data = network3.load_data_shared(filename = data_dir + 'small_cats_and_dogs_data_400_100_100.pkl.gz')
#training_data, validation_data, test_data = network3.load_data_shared(filename = data_dir + 'small_cats_and_dogs_data.pkl.gz')

# set parameters
mini_batch_size = 100
epochs = 1
lmbda = 0.1

# before beginning processing, print the network architecture to standard output
print '''

Network Architecture:

   ConvPoolLayer(image_shape=(mini_batch_size, 1, 224, 224),
       filter_shape=(20, 1, 9, 9),
       poolsize=(2, 2),
       activation_fn=ReLU)

   ConvPoolLayer(image_shape=(mini_batch_size, 20, 108, 108),
       filter_shape=(20, 20, 9, 9),
       poolsize=(2, 2),
       activation_fn=ReLU)

   FullyConnectedLayer(n_in=20*50*50, n_out=1000, activation_fn=ReLU)

   SoftmaxLayer(n_in=1000, n_out=2)

'''

# also print the training hyperparamters
print 'learning rate: ' + str(eta)
print 'regularization parameter: ' + str(lmbda)
print 'mini-batch size: ' + str(mini_batch_size)
print 'epochs: ' + str(epochs)
print

# define the network architecture
layers = [
   ConvPoolLayer(image_shape=(mini_batch_size, 1, 224, 224),
       filter_shape=(20, 1, 9, 9),
       poolsize=(2, 2),
       activation_fn=ReLU),
   ConvPoolLayer(image_shape=(mini_batch_size, 20, 108, 108),
       filter_shape=(20, 20, 9, 9),
       poolsize=(2, 2),
       activation_fn=ReLU),
   FullyConnectedLayer(n_in=20*50*50, n_out=1000, activation_fn=ReLU),
   SoftmaxLayer(n_in=1000, n_out=2)]

# initialize and train the network
net = Network(layers, mini_batch_size)
net.SGD(training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=lmbda, save_model_as=save_model_as)
