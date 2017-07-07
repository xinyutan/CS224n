#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
	"""
	Forward and backward propagation for a two-layer sigmoidal network

	Compute the forward propagation and for the cross entropy cost,
	and backward propagation for the gradients for all parameters.

	Arguments:
	data -- M x Dx matrix, where each row is a training example.
	labels -- M x Dy matrix, where each row is a one-hot vector.
	params -- Model parameters, these are unpacked for you.
	dimensions -- A tuple of input dimension, number of hidden units
				  and output dimension
	"""

	### Unpack network parameters (do not modify)
	ofs = 0
	Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

	W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
	ofs += Dx * H
	b1 = np.reshape(params[ofs:ofs + H], (1, H))
	ofs += H
	W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
	ofs += H * Dy
	b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

	### YOUR CODE HERE: forward propagation
	gradb1 = np.zeros_like(b1)
	gradW1 = np.zeros_like(W1)
	gradb2 = np.zeros_like(b2)
	gradW2 = np.zeros_like(W2)
	cost = 0
	for i in range(data.shape[0]):
		## forward pass
		datai = data[i].reshape((1, Dx))
		labeli = labels[i].reshape((1, Dy))
		
		h = sigmoid(np.matmul(datai, W1) + b1)
		yhati = softmax(np.matmul(h, W2) + b2)
		cost += -np.sum(labeli * np.log(yhati))
		
		## backpropagation
		grada2 = yhati - labeli
		gradb2 += grada2
		gradW2 += np.matmul(h.T, grada2)
		
		grada1 = np.matmul(grada2, W2.T)
		gradb1 += h * (1 - h) * grada1
		gradW1 += np.matmul(datai.T, h*(1-h)*grada1)
		
	### END YOUR CODE

	### Stack gradients (do not modify)
	grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
		gradW2.flatten(), gradb2.flatten()))
	print("grad: ", grad)
	return cost, grad


def sanity_check():
	"""
	Set up fake data and parameters for the neural network, and test using
	gradcheck.
	"""
	print ("Running sanity check...")

	N = 20
	dimensions = [10, 5, 10]
	data = np.random.randn(N, dimensions[0])   # each row will be a datum
	labels = np.zeros((N, dimensions[2]))
	for i in range(N):
		labels[i, random.randint(0,dimensions[2]-1)] = 1
	print(labels)
	params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
		dimensions[1] + 1) * dimensions[2], )

	gradcheck_naive(lambda params:
		forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
	"""
	Use this space add any additional sanity checks by running:
		python q2_neural.py
	This function will not be called by the autograder, nor will
	your additional tests be graded.
	"""
	print("Running your sanity checks...")
	### YOUR CODE HERE
	raise NotImplementedError
	### END YOUR CODE


if __name__ == "__main__":
	sanity_check()
	#your_sanity_checks()
