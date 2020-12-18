import tensorflow as tf
import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import PIL
from tensorflow.keras import layers


data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

# img = data_augmentation(img)

img = cv2.imread("./data/20176816/20176816_1.jpg")
img = data_augmentation(img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = tf.image.adjust_brightness(img, 0.9)
# # print(img)
# img = tf.make_ndarray(img)
# img = PIL.Image.open(img)
plt.imshow(img)
plt.show()