# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:16:53 2019

@author: Administrator
"""


import numpy as np
from DataReader import DataReader
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, BatchNormalization, Activation

def different_load_set(data, train_set, test_set):
    if train_set == 'A':
        x_train = data[0][:6600]
        y_train = data[3][:6600]
    elif train_set == 'B':
        x_train = data[0][6600: 13200]
        y_train = data[3][6600: 13200]
    elif train_set == 'C':
        x_train = data[0][13200:]
        y_train = data[3][13200:]
    
    if test_set == 'A':
        x_test = data[1][:250]
        x_valid = data[2][:70]
        y_test = data[4][:250]
        y_valid = data[5][:70]
    elif test_set == 'B':
        x_test = data[1][250: 500]
        x_valid = data[2][70: 140]
        y_test = data[4][250: 500]
        y_valid = data[5][70: 140]
    elif test_set == 'C':
        x_test = data[1][500:]
        x_valid = data[2][140:]
        y_test = data[4][500:]
        y_valid = data[5][140:]
    
    return x_train, x_test, x_valid, y_train, y_test, y_valid

if __name__ == '__main__':

    data_augment_stride = 64
    sample_size = 2048
    test_set_num = 25
    train_set_num = 660
    valid_set_num = 7
    
    path = ['.\data\A', '.\data\B', '.\data\C']
    
    x_train = np.zeros((1, sample_size, 2))
    x_test = np.zeros((1, sample_size, 2))
    x_valid = np.zeros((1, sample_size, 2))
    y_train = np.zeros((1))
    y_test = np.zeros((1))
    y_valid = np.zeros((1))
    
    data = [x_train, x_test, x_valid, y_train, y_test, y_valid]
    
    data_reader = DataReader(data_augment_stride=data_augment_stride, 
                             train_set_num=train_set_num)
    for i in range(len(path)):
        data_sub = data_reader.read_data(path[i])
        for i in range(len(data)):
            data[i] = np.append(data[i], data_sub[i], axis = 0)
    
    for i in range(len(data)):
        data[i] = data[i][1:]
    # train set A, test set B
    x_train, x_test, x_valid, y_train, y_test, y_valid = different_load_set(data, 'B', 'A')
    # for t_SNE
    y_label = y_test
    
    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)
    y_valid = to_categorical(y_valid, num_classes = 10)
    
    #-----------------构建网络-------------------------------------------------
   
    model = Sequential()
    
    model.add(BatchNormalization(input_shape = (sample_size, 2)))
    #Convolution1
    model.add(Conv1D(filters = 16,
                     kernel_size = 64,
                     strides = 16,
                     padding = 'same',
                     kernel_initializer = 'he_normal'))
    
    model.add(BatchNormalization())
    
    model.add(Activation('relu'))
    
    model.add(MaxPool1D(pool_size = 2))
    #Convolution2
    model.add(Conv1D(filters = 32,
                     kernel_size = 3,
                     strides = 1,
                     padding = 'same',
                     kernel_initializer = 'he_normal'))
    
    model.add(BatchNormalization())
    
    model.add(Activation('relu'))
    
    model.add(MaxPool1D(pool_size = 2))
    #Convolution3
    model.add(Conv1D(filters = 64,
                     kernel_size = 3,
                     strides = 1,
                     padding = 'same',
                     kernel_initializer = 'he_normal'))
    
    model.add(BatchNormalization())
    
    model.add(Activation('relu'))
    
    model.add(MaxPool1D(pool_size = 2))
    #Convolution4
    model.add(Conv1D(filters = 64,
                     kernel_size = 3,
                     strides = 1,
                     padding = 'same',
                     kernel_initializer = 'he_normal'))
    
    model.add(BatchNormalization())
    
    model.add(Activation('relu'))
    
    model.add(MaxPool1D(pool_size = 2))
    #Convolution5
    model.add(Conv1D(filters = 64,
                     kernel_size = 3,
                     strides = 1,
                     kernel_initializer = 'he_normal'))
    
    model.add(BatchNormalization())
    
    model.add(Activation('relu'))
    
    model.add(MaxPool1D(pool_size = 2))
    #Flatten
    model.add(Flatten())
    
    model.add(Dense(units = 100))
    
    model.add(BatchNormalization())
    
    model.add(Activation('relu'))
    
    model.add(Dense(units = 10,
                    activation = 'softmax'))
    #Train
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    history = model.fit(x_train, 
                        y_train, 
                        batch_size = 256, 
                        epochs = 2000, 
                        validation_data = (x_valid, y_valid))
    
    score = model.evaluate(x_test, y_test)
