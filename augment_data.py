import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob

from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

def my_preprocessing_func(img):
    image = np.array(img)
    return image / 255

def find_label(class_dict, need_find):
    for key, value in class_dict.items():
        if value == need_find:
            return key

new_data_dir = './aug_data/'
if not os.path.exists(new_data_dir):
    os.makedirs(new_data_dir)

datagen = ImageDataGenerator(rotation_range=15, fill_mode='nearest', brightness_range=[0.4,1.5], horizontal_flip=True)
img = datagen.flow_from_directory('./data/', batch_size=16, target_size=(160, 160), shuffle = True)
class_dictionary = img.class_indices

count = 0
batch = next(img)
while(batch):
    if(count > 3699):
        break
    for i in range(len(batch[0])):
        image = batch[0][i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        real_label = find_label(class_dictionary, np.argmax(batch[1][i]))
        save_dir = new_data_dir + real_label + '/'
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        img_name = real_label + "_" + str(count) + ".jpg"
        img_save_path = save_dir + img_name
        cv2.imwrite(img_save_path, image) 
        count += 1
    batch = next(img)
print("Done gen {} images".format(count))

dir = glob.glob("./aug_data/*")

count = 0
for dir_ in dir:
    dir_ = dir_ + "/*"
    img = glob.glob(dir_)
    for img_ in img:
        count += 1
    dir_name = dir_.split("/")[-2]
    print("Person {} have {} images".format(dir_name, count))
    count = 0



