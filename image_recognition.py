import numpy as np
import os
import cv2
import pickle
import argparse

from mtcnn import MTCNN
from glob import glob
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from src.utils import *
# from mtcnn import MTCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ap = argparse.ArgumentParser()

ap.add_argument("--path", default="", help="Path to image")
ap.add_argument("--facedetect", default="no", help="Parameter to detect face")
args = ap.parse_args()

#threshold
softmax_thresh = 0.96
cosine_thresh = 0.80

#load images to predict
img_test = cv2.imread(args.path)

#load face model
face_model_path = './models/facenet_keras.h5'
face_model = load_model(face_model_path)

#read detector
detector = MTCNN()

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

if(args.facedetect == "yes"):
    img = img_test
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                # print(embed)

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
    cv2.imshow('Image',img)
    cv2.waitKey() == ord('q')

    #save outputs
    output = "./output/" + args.path.split('/')[-1]
    cv2.imwrite(output, img) 
else:
    #resize
    img_test = cv2.resize(img_test, (160, 160))

    #preprocess img and embedding
    img_test = prewhiten(img_test)
    img_test = img_test[np.newaxis]
    # embed = face_model.predict_on_batch(img_test)
    embed = calc_embs(face_model, img_test, 1)

    #predict with softmax model
    preds = classify_model.predict(embed)
    preds = preds.flatten()
    j = np.argmax(preds)
    proba = preds[j]
    print("Softmax method:")
    print("Prob: {:.3f}".format(proba))
    a_label = [1 if n == proba else 0 for n in preds]
    print("Label predict: ", lb.classes_[j])

    #predict with cosine distance
    label, prob = most_similarity(np.concatenate(dict_embs["embs"]), embed, dict_embs["labels"])
    print("\nCos method:")
    print("Prob: {:.3f}".format(proba))
    print("Label predict: ", label)


    