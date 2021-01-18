# Face Recognition - Realtime Recognition
## Table of Contents  
#### [1. Introduction](#headers)
#### [2. Requirement](#requirement)   
#### [3. Usage](#usage) 
#### [4. Reference](#reference) 
<br>
<a name="headers"/>

### 1. Introduction
- Face recognition is a technology capable of matching a human face from a digital image or a video frame against a database of faces, typically employed to authenticate users through ID verification services, works by pinpointing and measuring facial features from a given image.
- Recognize and manipulate faces with Python and its support libraries.
The project uses MTCNN for detecting faces, then applies a simple alignment for each detected face and feeds those aligned faces into embeddings model (Facenet).
Finally, a softmax classifier was put on top of embedded vectors for classification task.

<a name="requirement"/>

### 2. Requirement
- Tensorflow-gpu (2.1.0)
- Python 3.7+
- Keras 2.2.4
- OpenCV-python (4.4.0)
- Pip or anaconda
#### Install MTCNN:
```
pip install mtcnn (for pip)
conda install -c conda-forge mtcnn (for conda)
```

<a name="usage"/>

### 3. Usage
#### Crawl data:
```
python src/save_data.py --name (name of save_dir)
```
> *Make sure your computer has a webcam*
#### Preprocessing:
- Train embedding model with your dataset, you need to organize dataset as follows:
> *Each person needs 4-5 raw photos with many different angle shooting: front, left, right, ...*
```
Face-Recognition
└───data/
      └───person1/
      |      └───person1_1.jpg
      |          person1_2.jpg
      |          .....
      └───person2/
      |      └───person2_1.jpg
      |          person2_2.jpg
      |          .....
      └───personN/
             └───personN_1.jpg
                 personN_2.jpg
                 .....
```
#### Augment data:
- After having the raw image files, run *augment_data.py* to augment more images, each person after running will have 100 different images. Augments: *rotation_range = 15, brightness_range=[0.4,1.5], horizontal_flip*
```
python augment_data.py
```
#### Train embedding model:
```
python train_embs.py
```
> *After embedding, embedded file will be saved to output / train_embs.pickle*
#### Train softmax model:
```
python train_classify.py
```
> *The number of classes i'm setting here is 37 classes, so change the number of classes that match your dataset*
#### Enjoy result:
<a href="https://github.com/manhminno/Face-Recognition/blob/master/output/1.jpg">
      <img alt="Qries" src="https://github.com/manhminno/Face-Recognition/blob/master/output/1.jpg">
</a>

*Label is name of saved-dir - box is green, unknow will don't have label - box is red. Here label is id of person.*
- For image recognition:
```
python image_recognition.py --path (path to image) --facedetect (use MTCNN to detect face before recognition - yes/no)
```
> *Output will be saved to output/(name_of_img.jpg)*
- For video recognition:
```
python video_recognition.py --path (path to video) --facemodel (path of facenet weights) --classifymodel (path of classify model weight) --embspath (path of embed dir)
```
- For stream recognition:
```
python stream_recognition.py
```
> *Make sure your computer has a webcam, here i'm setting webcam 0*

<a name="reference"/>

### 4. Reference
- <strong><a href="https://github.com/deepinsight/insightface">Insightface</a></strong>
- <strong><a href="https://github.com/davidsandberg/facenet">Facenet</a></strong>
- <strong><a href="https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py">MTCNN</a></strong>



