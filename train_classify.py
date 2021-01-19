import keras
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

#build softmax model to classify
class SoftMax():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        return model

#Model output: 128-D, classes = 38
model = SoftMax(input_shape=(128,), num_classes=38)
model = model.build()
# model.summary()

#Load embedding vector
file = open('./output/train_embs.pickle', 'rb')
dict_embs = pickle.load(file)
embeddings = np.concatenate(dict_embs["embs"])

#Label2id
lb = LabelBinarizer()
labels = lb.fit_transform(dict_embs["labels"])

#create KFold to split dataset
cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

#train
for train_idx, valid_idx in cv.split(embeddings):
    X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
    his = model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(X_val, y_val))
    print(his.history['accuracy'])
    history['acc'] += his.history['accuracy']
    history['val_acc'] += his.history['val_accuracy']
    history['loss'] += his.history['loss']
    history['val_loss'] += his.history['val_loss']

model.save("./models/classify_model.h5")

#plot
#summary history for acc
plt.figure(figsize=(8, 9))
plt.subplot(211)
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')

#summary history for loss
plt.subplot(212)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./output/accuracy_loss.png')
plt.show()
