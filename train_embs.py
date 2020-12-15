import numpy as np
import os
import cv2
import pickle
from src.utils import *
from glob import glob
from keras.models import load_model
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Load model
model_path = './models/facenet_keras.h5'
model = load_model(model_path)

train_path = './data/'
dir = glob(train_path + '*' )

#init
labels = []
embs = []
img_list = {}
a_person = []

#make dict have key is person and value is images
for person_dir in dir:
    name = person_dir.split("\\")[-1]
    person_dir  = person_dir + "/*"
    imgs = glob(person_dir)
    for img in imgs:
        img_read = cv2.imread(img)
        img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        img_read = cv2.resize(img_read, (160, 160))
        a_person.append(img_read)
    img_list[name] = np.array(a_person)
    a_person = []

#calculate embedding vector
for name, imgs in tqdm(img_list.items()):
    embs_ = calc_embs(model, imgs, 1)    
    labels.extend([name] * len(embs_))
    embs.append(embs_)
    pass

#save to file pickle
dict = {}
dict["labels"] = labels
dict["embs"] = embs
f = open("./output/train_embs.pickle", "wb")
f.write(pickle.dumps(dict))
f.close()

print("Embedding done!")