#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import keras
import json
import cv2

def read_mnist():
    ims_test = np.fromfile('./mnist/t10k-images.idx3-ubyte', dtype='ubyte')
    iml_test = np.fromfile('./mnist/t10k-labels.idx1-ubyte', dtype='ubyte')
    
    ims_train = np.fromfile('./mnist/train-images.idx3-ubyte', dtype='ubyte')
    iml_train = np.fromfile('./mnist/train-labels.idx1-ubyte', dtype='ubyte')
    
    iml_test_head = iml_test[:8].view('>i4')
    iml_test_data = iml_test[8:]
    
    iml_train_head = iml_train[:8].view('>i4')
    iml_train_data = iml_train[8:]
    
    ims_test_head = ims_test[:16].view('>i4')
    ims_test_data = ims_test[16:].reshape(ims_test_head[1:])
    
    ims_train_head = ims_train[:16].view('>i4')
    ims_train_data = ims_train[16:].reshape(ims_train_head[1:])
    
    test_x = np.array([cv2.Canny(x, 100, 200) for x in ims_test_data])
    train_x = np.array([cv2.Canny(x, 100, 200) for x in ims_train_data])
    
    
    return {
            'test_X': test_x[..., np.newaxis],
            'test_Y': keras.utils.to_categorical(iml_test_data),
            'train_X': train_x[..., np.newaxis],
            'train_Y': keras.utils.to_categorical(iml_train_data)
            }
    
#model = keras.Sequential()
#model.add(keras.layers.Conv2D(64, kernel_size=5, input_shape=(28,28,1),
#                              activation='softmax'))
#model.add(keras.layers.Conv2D(32, kernel_size=3, activation='softmax'))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(10, activation='softmax'))
    
model = keras.Sequential()
model.add(keras.layers.Conv2D(10, kernel_size=3, input_shape=(28,28,1),
                              activation='hard_sigmoid'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(10, kernel_size=3, activation='hard_sigmoid'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))
#model.add(keras.layers.Dense(10, activation='sigmoid'))

def store(fn):
    with open(fn, 'wt') as f:
        d = model.get_weights()
        ld = [x.tolist() for x in d]
        json.dump(ld, f)
        
def load(fn):
    with open(fn, 'rt') as f:
        ld = json.load(f)
        d = [np.array(x) for x in ld]
        model.set_weights(d)

def train():
    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.1)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    data = read_mnist()
    return model.fit(data['train_X'], data['train_Y'], epochs=3)
    

def test():
    data = read_mnist()
    ans = model.predict(data['test_X'])
    n = np.argmax(ans, axis=1) == np.argmax(data['test_Y'], axis=1)
    return n.sum() / len(n)
    
def predict(img):
    img = img.reshape((1,28,28,1))
    ans = model.predict(img)[0]
    return np.max(ans), np.argmax(ans), ans