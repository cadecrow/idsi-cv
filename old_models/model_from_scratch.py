#!/usr/bin/env python
# coding: utf-8
from pre_processing import load_images as l_i
from pre_processing import load_file as l_f
# from pre_processing import mini_batch as m_b

from PIL import Image
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import random

np.random.seed(1)
data_file_name = "full_data_post.csv"
path_training = "data/train/" # path from current folder to training images
path_testing = "data/test/" # path from current folder to testing images

###Train Data#######################################################################
### Collect total number of training examples
m = l_f.check_num_images(data_file_name)
m_training = 1000
m_testing = 1000

### Useful for mini-batches
# num_of_training_batches = 3
# num_of_testing_batches = 1
### Divide the training examples into random mini-batches
# (training_batches, testing_batches) = m_b.mini_batch_rows(num_of_training_batches, num_of_testing_batches, m)

### Take a random thousand examples
list_of_rows = np.arange(1, m+1, 1)
np.random.shuffle(list_of_rows)
### Load the data accordingly
img_dict = l_f.load_file(data_file_name, list_of_rows[:m_training])
### Load and process the images
(train_x, train_y) = l_i.load_images(path_training, img_dict)
print(train_x.shape)

### Reshape train_y to one-hot vectors
id_matrix = np.identity(4)
train_y_matrix = id_matrix[train_y - 1]
print(train_y_matrix.shape)

###Test Data#######################################################################
### Load the data accordingly
img_dict = l_f.load_file(data_file_name, list_of_rows[m_training:m_testing])
### Load and process the images
(test_x, test_y) = l_i.load_images(path_testing, img_dict)
print(test_x.shape)

### Reshape test_y to one-hot vectors
test_y_matrix = id_matrix[test_y - 1]
print(test_y_matrix.shape)

###Defining the sigmoid and relu functions#######################################################################

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ


###Initialization#######################################################################

def initialize_parameters_deep(layer_dims):

	np.random.seed(3)
	parameters = {}
	L = len(layer_dims) # number of layers in the network

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

	assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
	assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

	return parameters


print(initialize_parameters_deep([5,4,3]))

###Forward propagation with multiple Relus and 1 sigmoid#######################################################################
def linear_forward(A, W, b):
	Z = np.dot(W, A) + b
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)

	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)

	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache


def L_model_forward(X, parameters):
	caches = []
	A = X
	L = len(parameters) // 2 # number of layers in the neural network

	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')

		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
	caches.append(cache)

	assert(AL.shape == (1,X.shape[1]))
	return AL, caches


###Computing Cost#######################################################################

def compute_cost(AL, Y):
	m = Y.shape[1]

	cost = (-1 / m) * (np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), (1-Y))))
	cost = np.squeeze(cost)

	assert(cost.shape == ())
	return cost

###Backward Propagation#######################################################################
def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = (1/m) * np.dot(dZ, A_prev.T)
	db = (1/m) * np.sum(dZ, axis=1, keepdims = True)
	dA_prev = np.dot(W.T, dZ)

	assert(dA_prev.shape == A_prev.shape)
	assert(dW.shape == W.shape)
	assert(db.shape == b.shape)

	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):

	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)], current_cache, 'relu')
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads


###Updating Paramaters#######################################################################

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2 # number of layers in the neural network

	for l in range(L):
		parameters["W" + str(l+1)] = parameters['W' + str(l+1)] - (learning_rate * grads['dW' + str(l+1)])
		parameters["b" + str(l+1)] = parameters['b' + str(l+1)] - (learning_rate * grads['db' + str(l+1)])

	return parameters


###Putting it together#######################################################################

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

	np.random.seed(1)
	costs = []

	parameters = initialize_parameters_deep(layers_dims)

	for i in range(0, num_iterations):
		AL, caches = L_model_forward(X, parameters)
		cost = compute_cost(AL, Y)
		grads = L_model_backward(AL, Y, caches)

		parameters = update_parameters(parameters, grads, learning_rate)

		# Print the cost every 100 training example
		if print_cost and i % 100 == 0:
			print("Cost after iteration %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)

	# plot the cost
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	return parameters


###Running the Model#######################################################################
layers_dims = [12288, 20, 5, 1] #3-layer model

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
