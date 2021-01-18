from mtcnn import MTCNN
import numpy as np
import os
import cv2
import glob
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
detector = MTCNN()

def move_dir():
    dir = glob.glob("./data2/class_data/*")    
    count = 0

    for img in dir:
        count += 1
        student = img.split('\\')[-1].split("_")[0]
        new_dir = "./data22/" + student + "/"
        if(count == 1):
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
        shutil.move(img, new_dir)
        print(new_dir)
        if(count == 5):
            count = 0

move_dir()

data = glob.glob("./data22/*")
for dir in data:
    # print(dir)
    img = glob.glob(dir + "/*")
    for a_img in img:
        # print(a_img)
        img_read = cv2.imread(a_img)
        img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        bboxes = detector.detect_faces(img_read)
        if len(bboxes) != 0:
            for bboxe in bboxes:
                bbox = bboxe['box']
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])  
                img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)         
                img2 = img_read[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                img2 = cv2.resize(img2, (160, 160))
                cv2.imwrite(a_img, img2)
        else:
            continue

    