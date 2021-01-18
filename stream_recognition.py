import numpy as np
import os
import cv2
import pickle
import glob 
from keras.models import load_model
from mtcnn import MTCNN
from sklearn.preprocessing import LabelBinarizer
from src.utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#read detector
detector = MTCNN()

#load model emb
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

#threshold
softmax_thresh = 0.96
cosine_thresh = 0.8

#load webcam
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = detector.detect_faces(img)

    # print(bboxes)
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
                img2 = prewhiten(img2)
                img2 = img2[np.newaxis]
                embed = face_model.predict_on_batch(img2)         
                
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
    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
