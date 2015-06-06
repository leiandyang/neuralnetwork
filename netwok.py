"""
network.py

"""
####Libraries
#Standard library
import random

#Third-party libraries
import numpy as np

class Network():
	
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.rand(y, x) 
				for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		"""Return the output of the network if ''a'' is input."""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid_vec(np.dot(w, a) + b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = NONE):
		"""Train the neural network using mini-batch stochastic gradient descent. The ''training_data'' is a list of tuples ''(x, y)'' representing the training inputs and the desired outputs. """
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(
					j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)
