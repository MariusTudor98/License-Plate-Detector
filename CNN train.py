# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:15:01 2021

@author: Marius-Iulian TUDOR
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

print('TensorFlow version =',tf.__version__)


path1=r'D:/licenta/data/'
path_v=r'D:/licenta/validare/'
width=28
height=28
channels=3
classes = 36
directories = [x[0] for x in os.walk('data')][1:]
log_dir=r'D:\licenta'
print(directories)
list=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
print (len(list))

def load_data():
    labels=[]
    data=[]
    for i in range(classes):
      
      path2 = path1 + str(list[i])
      images = os.listdir(path2)
      for img in images:
          image=cv2.imread(path2+ '/' + img)
          image_fromarray = Image.fromarray(image, "RGB")
          resize_image = image_fromarray.resize((height, width))
          data.append(np.array(resize_image))
          labels.append(i)
    labels=np.array(labels)
    data=np.array(data)
    print('loaded')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)
    return (X_train, y_train), (X_test, y_test)

def load_data2():
    labels=[]
    data=[]
    for i in range(classes):
      
      path2 = path_v + str(list[i])
      images = os.listdir(path2)
      for img in images:
          image=cv2.imread(path2+ '/' + img)
          image_fromarray = Image.fromarray(image, "RGB")
          resize_image = image_fromarray.resize((height, width))
          data.append(np.array(resize_image))
          labels.append(i)
    labels=np.array(labels)
    data=np.array(data)
    print('loaded')
    X_validate=data
    y_validate=labels
    return (X_validate, y_validate)

def create_model():
    model=models.Sequential()
    model.add(layers.Conv2D(28, (3,3), activation='relu',input_shape=(height,width,channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(56, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(56, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(56, activation='relu'))
    model.add(layers.Dense(36, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    history=model.fit(train_images, train_labels,validation_data=valid_set, epochs=100, callbacks=tensorboard_callback)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    val_loss, val_accuracy=model.evaluate(validate_images, validate_labels)
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acuratetea obtinuta')
    plt.ylabel('Acuratetea')
    plt.xlabel('Epoca')
    plt.legend(['antrenare', 'validare'], loc='lower right')
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Eroarea')
    plt.ylabel('Eroarea')
    plt.xlabel('Epoca')
    plt.legend(['antrenare', 'validare'], loc='upper right')
    plt.tight_layout()
    print(test_acc)
    
    model.save("model_char_recognition.h5")
    return model
       

#incarcare date si preprocesare
(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images.reshape((train_images.shape[0], height, width, channels))
test_images = test_images.reshape((test_images.shape[0], height, width,channels))
train_images = train_images / 255.0
test_images =  test_images / 255.0
(validate_images, validate_labels)=load_data2()
validate_images = validate_images.reshape((validate_images.shape[0], height, width, channels))
validate_images =  validate_images / 255.0
valid_set=(validate_images, validate_labels)
create_model()

