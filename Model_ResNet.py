import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


import keras.backend as K

from Model_ResNet_helper import *
from preprocessing import load_images as l_i
from preprocessing import load_file as l_f

from PIL import Image
from os import listdir
import random

np.random.seed(1)
#data_file_name = "../sampled_data.csv"
data_file_name = "sampled_data.csv"
#path_training = "../train/images/" # path from current folder to training images
path_training = "train/images/" # path from current folder to training images
#path_testing = "data/test/" # path from current folder to testing images
#path_testing = "../train/images/" # path from current folder to testing images
path_testing = "train/images/" # path from current folder to testing images

### Set training and testing details #################################################
m = l_f.check_num_images(data_file_name)
m_training = 20000
m_testing = 1000
num_epochs = 20
num_batches = 32


### Train Data #######################################################################
### Shuffle the m examples
list_of_rows = np.arange(1, m+1, 1)
np.random.shuffle(list_of_rows)
### Take a random m_training examples and load the data
img_dict = l_f.load_file(data_file_name, list_of_rows[:m_training])
### Load and process the images
(train_x, train_y) = l_i.load_images(path_training, img_dict)

### Normalize image vectors
train_x = train_x / 255

### Convert to one hot vector
print(train_x.shape)
print(train_y.shape)


### Test Data #######################################################################
### Take a random m_testing examples and load the data
#img_dict = l_f.load_file(data_file_name, list_of_rows[m_training:m_testing])
img_dict = l_f.load_file(data_file_name, list_of_rows[m_training:m_training + m_testing])
### Load and process the images
(test_x, test_y) = l_i.load_images(path_testing, img_dict)
#print(test_y)

### Normalize image vectors
test_x = test_x / 255

### Convert to one hot vector
print(test_x.shape)
print(test_y.shape)

#model = ResNet50(input_shape = (64, 64, 3), classes = 1) #changed classes from 4 to 2
model = ResNet50(input_shape = (64, 64, 3)) #changed classes from 4 to 2
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs = num_epochs, batch_size = num_batches)

preds = model.evaluate(test_x, test_y)
print(preds)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
