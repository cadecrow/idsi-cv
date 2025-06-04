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
m_training = 200 #20000
m_testing = 100 #1000
num_epochs = 3  #50
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


## To save the weights


# serialize model to JSON
model_json = model.to_json()
with open("model_uptodate.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_uptodate_weights.h5")
print("Saved model to disk")

# How to load the saved pre-trained weights
# load json and create model
# json_file = open('model_uptodate.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model_uptodate_weights.h5")
# print("Loaded model from disk")

# loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Note: remember to run line 106 to recompile the loaded model 


evaluation = model.evaluate(test_x, test_y)
print(evaluation)
print ("Loss = " + str(evaluation[0]))
print ("Test Accuracy = " + str(evaluation[1]))


import csv

## To compute only accuracy


# error_analysis_num = 1 # update so it doesn't overwrite previous csv files
# list_of_row_nums = list_of_rows[m_training:m_training + m_testing]
# preds = model.predict(test_x)

# with open("error_analysis" + str(error_analysis_num) + ".csv", "w", newline = "") as csv_write_file: 
#     line = csv.writer(csv_write_file)
#     headers = ["img_name", "uid", "subtype", "prediction"]
#     line.writerow(headers)
#     with open(data_file_name) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = 0
#         sample_count = 0
#         for row in csv_reader:
#             if (row[0] == ""):
#                 break
#             if line_count in list_of_row_nums:
#                 img_name = row[17]
#                 uid = row[20]
#                 subtype = row[19]
#                 prediction = str(int(round(preds[sample_count][0])))
#                 sample_count += 1
#                 if (prediction != subtype):
#                     # Save information to csv file
#                     new_entry = [img_name, uid, subtype, prediction]
#                     line.writerow(new_entry)
#             line_count += 1


## To compute precision and recall

error_analysis_num = 1 # update so it doesn't overwrite previous csv files
list_of_row_nums = list_of_rows[m_training:m_training + m_testing]
preds = model.predict(test_x)

TP = 0
TN = 0
FP = 0
FN = 0

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
                    # Update FP / FN
                    if (prediction == "1"):
                        FP += 1
                    else:
                        FN += 1
                else:
                    if (prediction == "1"):
                        TP += 1
                    else:
                        TN += 1
            line_count += 1

print("TP: " + str(TP))
print("TN: " + str(TN))
print("FP: " + str(FP))
print("FN: " + str(FN))

accuracy = round((TP + TN) / (TP + TN + FP + FN), 2)
print("Accuracy: " + str(accuracy))

precision = round(TP / (TP + FN), 2)
print("Precision: " + str(precision))

recall = round(TP / (TP + FP), 2)
print("Recall: " + str(recall))

F1 = (2 * precision * recall) / (precision + recall)
print("F1: " + str(F1))

