from PIL import Image
from preprocessing import crop_image as c_i
import numpy as np
from os import listdir

### Helper functions ####################
def test_human(path, img_dict):
    no_damage = [0,0]
    major_damage = [0,0]
    accuracy = (no_damage, major_damage)
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
            img_rgb.show()
            print("no_damage - minor_damage   = 0")
            print("major_damage - destroyed   = 1")
            human_guess = int(input("Enter damage rating: "))
            ### Take the Y value (already converted to number encoding: 0, 1)
            Y = int(img_dict[fname][uid][0])
            accuracy = compare_results(human_guess, Y, accuracy)
    return accuracy

def compare_results(human_guess, Y, accuracy):
    if (human_guess == Y):
        accuracy[Y][0] += 1
    else:
        accuracy[Y][1] += 1
        print("You guessed " + str(human_guess) + " but the correct answer was " + str(Y))
    return accuracy
