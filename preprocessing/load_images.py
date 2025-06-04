from PIL import Image
from preprocessing import crop_image as c_i
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
    images_y = np.array(images_y).T
    return (images_x, images_y)
