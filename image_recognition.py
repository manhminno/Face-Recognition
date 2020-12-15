import numpy as np
import os
import cv2
import pickle
from glob import glob

from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from src.utils import *
# from mtcnn import MTCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#load face model
face_model_path = './models/facenet_keras.h5'
face_model = load_model(face_model_path)

#load classify model
classify_model_path = './models/classify_model.h5'
classify_model = load_model(classify_model_path)

#load dict
dict_file = open('./output/train_embs.pickle', 'rb')
dict_embs = pickle.load(dict_file)
labels = dict_embs["labels"]

#label2id
lb = LabelBinarizer()
labels = lb.fit_transform(dict_embs["labels"])

# #Read detector
img_test = cv2.imread('../newtrain/manhnv/manhnv_10.jpg') #path to img to test
# img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

img_test = cv2.resize(img_test, (160, 160))

#preprocess img and embedding
img_test = prewhiten(img_test)
img_test = img_test[np.newaxis]
embed = face_model.predict_on_batch(img_test)

#predict with softmax model
preds = classify_model.predict(embed)
preds = preds.flatten()
j = np.argmax(preds)
proba = preds[j]
print("Softmax method:")
print("Prob: ", proba)
a_label = [1 if n == proba else 0 for n in preds]
print("Label predict: ", lb.classes_[j])

#predict with cosine distance
label, prob = most_similarity(np.concatenate(dict_embs["embs"]), embed, dict_embs["labels"])
print("\nCos method:")
print("Prob: ", str(prob[0]))
print("Label predict: ", label)


    