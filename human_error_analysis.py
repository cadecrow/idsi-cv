from human_error_analysis_helper import *
from preprocessing import load_file as l_f
import numpy as np

np.random.seed(1)
data_file_name = "full_data_train_post.csv"
path_training = "train/images/" # path from current folder to training images

### Collect total number of training examples
m = l_f.check_num_images(data_file_name)
m_training = 100

### Shuffle the m examples
list_of_rows = np.arange(1, m+1, 1)
np.random.shuffle(list_of_rows)
### Take a random m_training examples and load the data
img_dict = l_f.load_file(data_file_name, list_of_rows[:m_training])

accuracy = test_human(path_training, img_dict)

if (accuracy[0][0] + accuracy[0][1]) != 0:
    print("no_damage - minor_damage  = " + str(round(accuracy[0][0]/(accuracy[0][0] + accuracy[0][1]), 2)) + "%")
    print(accuracy[0])
if (accuracy[1][0] + accuracy[1][1]) != 0:
    print("major_damage - destroyed = " + str(round(accuracy[1][0]/(accuracy[1][0] + accuracy[1][1]), 2)) + "%")
    print(accuracy[1])
