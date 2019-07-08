# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 09:51:02 2019

@author: Administrator
"""


import numpy as np
from DataReader import DataReader
from Preprocessor import Preprocessor
from keras.utils import to_categorical
from keras.models import Sequential
from keras.initializers import he_normal
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout

if __name__ == '__main__':

    data_augment_stride = 64
    sample_size = 2048
    test_set_num = 25
    train_set_num = 660
    valid_set_num = 7
    
    conv1_filters = 32
    conv1_kernel_size = 20
    conv1_strides = 8
    
    pool1_pool_size = 4
    
    conv2_filters = 64
    conv2_kernel_size = 5
    conv2_strides = 2
    
    pool2_pool_size = 2
    
    path = ['.\data\A', '.\data\B', '.\data\C']
    
    x_train = np.zeros((1, sample_size))
    x_test = np.zeros((1, sample_size))
    x_valid = np.zeros((1, sample_size))
    y_train = np.zeros((1))
    y_test = np.zeros((1))
    y_valid = np.zeros((1))
    
    data = [x_train, x_test, x_valid, y_train, y_test, y_valid]
    
    data_reader = DataReader()
    for i in range(len(path)):
        data_sub = data_reader.read_data(path[i])
        for i in range(len(data)):
            data[i] = np.append(data[i], data_sub[i], axis = 0)
    
    for i in range(len(data)):
        data[i] = data[i][1:]
        
    x_train = data[0]
    x_test = data[1]
    x_valid = data[2]
    y_train = data[3]
    y_test = data[4]
    y_valid = data[5]
    
    preprocessor = Preprocessor()
    x_train, x_test, x_valid = preprocessor.preprocess(x_train, x_test, x_valid)
    
    x_train = np.expand_dims(x_train, axis = 2)
    x_test = np.expand_dims(x_test, axis = 2)
    x_valid = np.expand_dims(x_valid, axis = 2)
    
    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)
    y_valid = to_categorical(y_valid, num_classes = 10)
    #-----------------构建网络-------------------------------------------------
    
    model = Sequential()
    
    model.add(Conv1D(input_shape = (sample_size, 1),
                     filters = conv1_filters,
                     kernel_size = conv1_kernel_size,
                     strides = conv1_strides,
                     padding = 'same',
                     activation = 'relu',
                     kernel_initializer = 'he_normal'))
    
    model.add(MaxPool1D(pool_size = pool1_pool_size))
    
    model.add(Conv1D(filters = conv2_filters,
                     kernel_size = conv2_kernel_size,
                     strides = conv2_strides,
                     padding = 'same',
                     activation = 'relu',
                     kernel_initializer = 'he_normal'))
    
    model.add(MaxPool1D(pool_size = pool2_pool_size))
    
    model.add(Flatten())
    
    model.add(Dropout(rate = 0.5))
    
    model.add(Dense(units = 500, activation = 'relu'))
    
    model.add(Dropout(rate = 0.5))
    
    model.add(Dense(units = 10,
                    activation = 'softmax'))
    
    adam = Adam()
    
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = adam,
                  metrics = ['accuracy'])
    
    history = model.fit(x_train, 
                        y_train, 
                        batch_size = 256, 
                        epochs = 2000, 
                        validation_data = (x_valid, y_valid))
    
    score = model.evaluate(x_test, y_test)
    