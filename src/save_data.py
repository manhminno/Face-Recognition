import numpy as np
import os
import cv2
import argparse
from mtcnn import MTCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ap = argparse.ArgumentParser()

ap.add_argument("--name", default="unknow", help="Name of person")
args = ap.parse_args()

#Read detector
detector = MTCNN()

#Load webcam
cap = cv2.VideoCapture(0)
img2 = None
count_frame = 0
count_img = 0
path_to_save = './newtrain/'
path_to_save = path_to_save + args.name + '/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
while 1:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = detector.detect_faces(img)

    # print(bboxes)
    if len(bboxes) != 0:
        for bboxe in bboxes:
            bbox = bboxe['box']
            bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])            
            
            #Box
            img2 = img[bbox[1]-40:bbox[3]+13, bbox[0]-15:bbox[2]+25]
            cv2.rectangle(img, (bbox[0]-20, bbox[1]-50), (bbox[2]+30, bbox[3]+15), (255,0,0), 2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Webcam',img)
    k = cv2.waitKey(24) & 0xff
    if k == ord('q'):
        break
    if(img2 is not None):
        count_frame += 1
        if(count_frame%4 == 0):
            save_img = path_to_save + args.name + "_" + str(count_img) + ".jpg"
            print("Image saved at: ", save_img)
            # cv2.imwrite(save_img, img2)   #Write data

            cv2.imshow('Image', img2)
            k = cv2.waitKey(24) & 0xff
            if k == ord('q'):
                break
            count_img += 1
            if(count_img == 20):
                break

cap.release()
cv2.destroyAllWindows()
exit(0)
