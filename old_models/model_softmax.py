#!/usr/bin/env python
# coding: utf-8
from pre_processing import load_images as l_i
from pre_processing import load_file as l_f


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.python.framework import ops
from PIL import Image
import numpy as np
from os import listdir, environ
import matplotlib.pyplot as plt

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# np.random.seed(1)
# data_file_name = "full_data_train_post.csv"
# path_training = "data/train/" # path from current folder to training images
# path_testing = "data/test/" # path from current folder to testing images

# ### Train Data #######################################################################
# ### Collect total number of training examples
# m = l_f.check_num_images(data_file_name)
# m_training = 50
# m_testing = 50

# ### Shuffle the m examples
# list_of_rows = np.arange(1, m + 1, 1)
# np.random.shuffle(list_of_rows)


# ### Take a random m_training examples and load the data
# img_dict = l_f.load_file(data_file_name, list_of_rows[:m_training])

# print(len(img_dict))

# ### Load and process the images
# (train_x, train_y) = l_i.load_images(path_training, img_dict)
# print(train_x.shape)
# print(train_y.shape)

# ### Test Data #######################################################################
# ### Take a random m_testing examples and load the data

# img_dict = l_f.load_file(data_file_name, list_of_rows[m_training:m_training + m_testing])
# ### Load and process the images
# (test_x, test_y) = l_i.load_images(path_training, img_dict) #getting test example from the same training folder
# print(test_x.shape)
# print(test_y.shape)

# ### Initialization #######################################################################

# tf.compat.v1.disable_eager_execution()

# def one_hot_matrix(Y, num_labels):
# 	num_labels = tf.constant(num_labels, name = "num_labels")
# 	one_hot_matrix = tf.one_hot(Y, num_labels, axis=0)
# 	sess = tf.compat.v1.Session()
# 	one_hot = sess.run(one_hot_matrix)
# 	sess.close()
# 	return one_hot

# def create_placeholders(n_x, n_y):
# 	# Creates placeholders for X and Y with n_x (num of X featres) and n_y (num of labels)
# 	X = tf.placeholder(tf.float32, shape = (n_x, None), name = 'X') # None allows for any number of columns
# 	Y = tf.placeholder(tf.float32, shape = (n_y,None), name = 'Y')
# 	return X, Y

def initialize_parameters(layer_dims):
	tf.set_random_seed(1)
	parameters = {}
	L = len(layer_dims) # number of layers in the network
	for l in range(1, L):
		parameters['W' + str(l)] = tf.get_variable("W" + str(l), [layer_dims[l], layer_dims[l-1]], initializer = tf.random_normal_initializer(seed = 1))
		parameters['b' + str(l)] = tf.get_variable("b" + str(l), [layer_dims[l], 1], initializer = tf.zeros_initializer())

	return parameters

parameters = initialize_parameters([3, 2])

init = tf.global_variables_initializer

with tf.Session() as session: 
    session.run(init)
    print(session.run(parameters))
	

# ### Forward propagation with two Relus and 1 softmax #######################################################################
# def forward_propagation(X, parameters):
# 	W1 = parameters['W1']
# 	b1 = parameters['b1']
# 	W2 = parameters['W2']
# 	b2 = parameters['b2']
# 	W3 = parameters['W3']
# 	b3 = parameters['b3']

# 	Z1 = tf.add(tf.matmul(W1,X),b1)
# 	A1 = tf.nn.relu(Z1)
# 	Z2 = tf.add(tf.matmul(W2,A1),b2)
# 	A2 = tf.nn.relu(Z2)
# 	Z3 = tf.add(tf.matmul(W3,A2),b3)
# 	# Returns the last linear output (not including Softmax activation function)
# 	return Z3

# ### Compute Cost #######################################################################

# def compute_cost(Z3, Y):
# 	logits = tf.transpose(Z3)
# 	labels = tf.transpose(Y)

# 	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

# 	return cost

# ### Putting it together #######################################################################
# def model(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate = 0.0001,
# 		  num_epochs = 1500, print_cost = True):

# 	ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
# 	tf.set_random_seed(1)                             # to keep consistent results
# 	(n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
# 	n_y = Y_train.shape[0]                            # n_y : output size
# 	costs = []                                        # To keep track of the cost

# 	X, Y = create_placeholders(n_x, n_y)

# 	parameters = initialize_parameters(layer_dims)

# 	Z3 = forward_propagation(X, parameters)

# 	cost = compute_cost(Z3, Y)

# 	# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
# 	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# 	# Initialize all the variables
# 	init = tf.global_variables_initializer()

# 	# Start the session to compute the tensorflow graph
# 	with tf.Session() as sess:

# 		# Run the initialization
# 		sess.run(init)

# 		# Do the training loop
# 		for epoch in range(num_epochs):
# 			_ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

# 			# Print the cost every epoch
# 			if print_cost == True and epoch % 100 == 0:
# 				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
# 			if print_cost == True and epoch % 5 == 0:
# 				costs.append(epoch_cost)

# 		# plot the cost
# 		plt.plot(np.squeeze(costs))
# 		plt.ylabel('cost')
# 		plt.xlabel('iterations (per fives)')
# 		plt.title("Learning rate =" + str(learning_rate))
# 		plt.show()

# 		parameters = sess.run(parameters)
# 		print ("Parameters have been trained!")

# 		# Calculate the correct predictions
# 		correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

# 		# Calculate accuracy on the test set
# 		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 		print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
# 		print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

# 		return parameters


# ###Running the Model#######################################################################
# layers_dims = [12288, 20, 5, 1] #3-layer model

# ### Reshape train_y to one-hot vectors
# train_y_matrix = one_hot_matrix(train_y, 4)

# ### Reshape test_y to one-hot vectors
# test_y_matrix = one_hot_matrix(test_y, 4)

# parameters = model(train_x, train_y_matrix, test_x, test_y_matrix, layers_dims, 0.0001, 2500, True)
