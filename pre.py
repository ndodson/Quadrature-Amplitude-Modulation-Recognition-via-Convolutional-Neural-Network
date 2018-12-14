import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import keras.backend as K



train_data = '/../../..'
test_data = '/../../..'

def one_hot_label(img):             ##apply some one-hot encoding
    global ohl
    label = img.split('.')[0]
    if label == '4qam':
        ohl = np.array([0,0,0,0,1])
    elif label == '8qam':
        ohl = np.array([0,0,0,1,0])
    elif label == '16qam':
        ohl = np.array([0,0,1,0,0])
    elif label == '32qam':
        ohl = np.array([0,1,0,0,0])
    elif label == '64qam':
        ohl = np.array([1,0,0,0,0])
    return ohl

def train_data_with_label():                    ##prepare training data
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:

            img = cv2.resize(img, (64,64))
            train_images.append([np.array(img), one_hot_label(i)])
        else:
            print("image not loaded")
    shuffle(train_images)
    return train_images

def test_data_with_label():             ##prepare testing data

    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data,i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:

            img = cv2.resize(img, (64,64))
            test_images.append([np.array(img), one_hot_label(i)])
        else:
            print("image not loaded")
    shuffle(test_images)
    return test_images


def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

training_images = train_data_with_label()
testing_images = test_data_with_label()
tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,1)
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,64,64,1)
tst_lbl_data = np.array([i[1] for i in testing_images])




model = Sequential()


##pass through the network

model.add(InputLayer(input_shape=[64,64,1]))
model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='tanh'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=50,kernel_size=5,strides=1,padding='same',activation='tanh'))
model.add(MaxPool2D(pool_size=5,padding='same'))


model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same',activation='tanh'))
model.add(MaxPool2D(pool_size=5,padding='same'))


model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512,activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(5,input_shape=(5,),activation='softmax'))
optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=tr_img_data,y=tr_lbl_data,epochs=4,batch_size=32)
model.summary()

