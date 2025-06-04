#!/usr/bin/env python
# coding: utf-8

# # ResNet Model



import numpy as np
from keras.models import Model

import keras.backend as K

from Model_ResNet_helper import *
from preprocessing import load_images as l_i
from preprocessing import load_file as l_f

from PIL import Image
from os import listdir
import random



data_file_name = "sampled_data.csv"
path_training = "train/images/" # path from current folder to training images
#path_testing = "data/test/" # path from current folder to testing images
path_testing = "train/images/" # path from current folder to testing images





### Set training and testing details #################################################
m = l_f.check_num_images(data_file_name)
m_training = 50
m_testing = 10
num_epochs = 2 
num_batches = 2 #32



### Train Data #######################################################################
np.random.seed(1)
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
img_dict_test = l_f.load_file(data_file_name, list_of_rows[m_training:m_training + m_testing])
### Load and process the images
(test_x, test_y) = l_i.load_images(path_testing, img_dict_test)
#print(test_y)

### Normalize image vectors
test_x = test_x / 255

### Convert to one hot vector
print(test_x.shape)
print(test_y.shape)



model = ResNet50(input_shape = (64, 64, 3)) #changed classes from 4 to 2
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs = num_epochs, batch_size = num_batches)



evaluation = model.evaluate(test_x, test_y)
print(evaluation)
print ("Loss = " + str(evaluation[0]))
print ("Test Accuracy = " + str(evaluation[1]))



import csv

error_analysis_num = 1 # update so it doesn't overwrite previous csv files
list_of_row_nums = list_of_rows[m_training:m_training + m_testing]
preds = model.predict(test_x)

with open("error_analysis" + str(error_analysis_num) + ".csv", "w", newline = "") as csv_write_file: 
    line = csv.writer(csv_write_file)
    headers = ["img_name", "uid", "subtype", "prediction"]
    line.writerow(headers)
    with open(data_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        sample_count = 0
        for row in csv_reader:
            if (row[0] == ""):
                break
            if line_count in list_of_row_nums:
                img_name = row[17]
                uid = row[20]
                subtype = row[19]
                prediction = str(int(round(preds[sample_count][0])))
                sample_count += 1
                if (prediction != subtype):
                    # Save information to csv file
                    new_entry = [img_name, uid, subtype, prediction]
                    line.writerow(new_entry)
            line_count += 1




# # Basic Model


from pre_processing import load_file as l_f

np.random.seed(1)
data_file_name = "CS230/full_data_train_post.csv"
path_training = "CS230/data/train/" # path from current folder to training images
path_testing = "CS230/data/test/" # path from current folder to testing images



from PIL import Image
from pre_processing import crop_image as c_i
import numpy as np
from os import listdir

# Load and process the images for the mini-batch
def load_images(path, img_dict):
    ### Starts a list of images
    images_x = []
    images_y = []
    ### Loops through image dictionary, opens the image file and processes it
    for fname in img_dict:
        img = Image.open(path + fname)
        ### For each image, go through the specific buildings asked for
        for uid in img_dict[fname]:
            ### Collect the polygon
            xy = img_dict[fname][uid][1]
            ### Crop the image, reshape, and append
            img_crop = c_i.crop_image(img, xy)
            img_rgb = img_crop.convert('RGB')
            
            img_pixels = np.array(img_rgb)
            
            images_x.append(img_pixels)
            
            ### Take the Y value and convert to number encoding
            Y = img_dict[fname][uid][0]
            images_y.append(Y)
            
    images_x = np.array(images_x)
    images_y = np.array(images_y)
    return (images_x, images_y)



### Train Data #######################################################################
### Collect total number of training examples
m = l_f.check_num_images(data_file_name)
m_training = 50 #20000
m_testing = 10  #5000

### Shuffle the m examples
list_of_rows = np.arange(1, m + 1, 1)
np.random.shuffle(list_of_rows)




### Take a random m_training examples and load the data
img_dict = l_f.load_file(data_file_name, list_of_rows[:m_training])

print(len(img_dict))

### Load and process the images
(train_x, train_y) = load_images(path_training, img_dict)
print(train_x.shape)
print(train_y.shape)



class_names = ['No damage', 'Minor damage', 'Major damage', 'Destroyed']



plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_x[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_y[i]])
plt.show()




### Test Data #######################################################################
### Take a random m_testing examples and load the data

img_dict = l_f.load_file(data_file_name, list_of_rows[m_training:m_training + m_testing])
### Load and process the images
(test_x, test_y) = load_images(path_training, img_dict) #getting test example from the same training folder
print(test_x.shape)
print(test_y.shape)



train_y -= 1 
test_y -= 1




train_x = train_x / 255.0

test_x = test_x / 255.0



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64, 3)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(4)
])




model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



model.fit(train_x, train_y, epochs=200, batch_size = 1000)



test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\nTest accuracy:', test_acc)







