import numpy as np
import os
import cv2
import pickle
import glob 
import argparse
import tensorflow as tf

from keras.models import load_model
from mtcnn import MTCNN
from sklearn.preprocessing import LabelBinarizer
from src.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


ap = argparse.ArgumentParser()

ap.add_argument("--path", default="./video/video_test.mp4", help="Path to video")
ap.add_argument("--facemodel", default="./models/facenet_keras.h5", help="Path to face model weight")
ap.add_argument("--classifymodel", default="./models/classify_model.h5", help="Path to classify model weight")
ap.add_argument("--embspath", default="./output/train_embs.pickle", help="Path to embs dir")

args = ap.parse_args()

#read detector
detector = MTCNN(steps_threshold=[0.85, 0.85, 0.85])

#load model emb
face_model_path = args.facemodel
face_model = load_model(face_model_path)

#load classify model
classify_model_path = args.classifymodel
classify_model = load_model(classify_model_path)

#load dict
dict_file = open(args.embspath, 'rb')
dict_embs = pickle.load(dict_file)
labels = dict_embs["labels"]

#label2id
lb = LabelBinarizer()
labels = lb.fit_transform(dict_embs["labels"])

#threshold
softmax_thresh = 0.96
cosine_thresh = 0.7

#load video path
cap = cv2.VideoCapture(args.path)

# with tf.device('/device:gpu:0'):
frames = 0
while 1:
    ret, img = cap.read()
    frames += 1
    img = cv2.resize(img, (1024, 576))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if(frames%4 == 0):
        bboxes = detector.detect_faces(img)

        if len(bboxes) != 0:
            for bboxe in bboxes:
                bbox = bboxe['box']
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                
                #Box
                img2 = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                color_box = (255, 0, 0)
                cv2.rectangle(img, (bbox[0]-10, bbox[1]-10), (bbox[2]+10, bbox[3]+10), color_box, 2)

                try:
                    img2 = cv2.resize(img2, (160, 160))
                    img2 = img2[np.newaxis]
                    # embed = face_model.predict_on_batch(img2)
                    embed = calc_embs(face_model, img2, 1)         
                    
                    preds = classify_model.predict(embed)
                    preds = preds.flatten()
                    pred, prob = get_label_classify(preds, lb)
                    print("Softmax method: Prob: {:.3f} Label predict: {}".format(prob, pred))

                    label_cos, prob_cos = most_similarity(np.concatenate(dict_embs["embs"]), embed, dict_embs["labels"])
                    print("Cosine method: Prob: {:.3f} Label predict: {}".format(prob_cos, label_cos))
                    print("--------------------")

                    #show result if 2 method have same predict
                    if(prob > softmax_thresh and pred == label_cos and prob_cos > cosine_thresh):
                        #box of face
                        color_box = (0, 255, 0) #change color to green if face regco
                        cv2.rectangle(img, (bbox[0]-10, bbox[1]-10), (bbox[2]+10, bbox[3]+10), color_box, 2)
                        
                        #text process
                        text = str(pred)
                        # scale = (round((bbox[2]+10+1 - bbox[0]-10-1)*0.0025)/2)+0.5
                        scale = (round((bbox[2]+10+1 - bbox[0]-10-1)*0.003)/2)+0.5
                        t_thickness = int(round(scale + 0.5))
                        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, thickness=t_thickness)[0]
                        width_box = -(bbox[0]-10-1 - bbox[2]+10+1)
                        locate = int(round((width_box - t_size[0])/2))
                        
                        #box and text
                        cv2.rectangle(img, (bbox[0]-10-1, bbox[1]-t_size[1]-25), (bbox[2]+10+1, bbox[1]-10), color_box, -1)
                        cv2.putText(img, text, (bbox[0]-10-1+locate+11, y-4-5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness=t_thickness)
                    else:
                        continue
                except:
                    print("Error!\n")
            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Webcam',img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
