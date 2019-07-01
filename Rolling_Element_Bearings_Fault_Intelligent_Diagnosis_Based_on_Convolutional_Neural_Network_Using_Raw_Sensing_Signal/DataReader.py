# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:15:44 2019

@author: Administrator
"""

import os
import glob
import numpy as np
from scipy.io import loadmat

class DataReader:
    
    def __init__(self, 
                 data_augment_stride = 64,
                 sample_size = 2048,
                 test_set_num = 25,
                 train_set_num = 660):
        
        self.__data_augment_stride = data_augment_stride
        self.__sample_size = sample_size
        self.__test_set_num = test_set_num
        self.__train_set_num = train_set_num

    def read_data(self, 
                  path, 
                  sensor_place = 'DE'):
        #所需数据长度，从最后一个开始，往回追溯
        length = (self.__test_set_num * self.__sample_size + self.__sample_size + 
                  (self.__train_set_num - 1) * self.__data_augment_stride)
        
        cate = [path + '\\' + x for x in os.listdir(path) if 
                os.path.isdir(path + '\\' + x)]
        
        train_data = np.zeros((1, self.__sample_size))
        test_data = np.zeros((1, self.__sample_size))
        train_label = np.zeros((1, 1))
        test_label = np.zeros((1, 1))
        
        for idx, folder in enumerate(cate):
            #读某一类数据,一类数据仅一个文件
            for t in glob.glob(folder + '\*.mat'):
                print('reading the data:%s'%(t))
                mat = loadmat(t)
                key = [key for key in mat.keys() if sensor_place in key]
                data_temp = mat[key[0]][-length:]
            
            train_data = np.append(train_data, 
                                   self.__data_segmentation(
                                           data_temp[: length -\
                                                     self.__test_set_num *\
                                                     self.__sample_size]),
                                    axis = 0)
            
            test_data = np.append(test_data,
                                  self.__data_segmentation(
                                          data_temp[length -\
                                                    self.__test_set_num *\
                                                    self.__sample_size:], False),
                                    axis = 0)
            
            train_label = np.append(train_label, 
                                    np.array([idx for _ in range(self.__train_set_num)]).reshape(-1, 1),
                                    axis = 0)
            
            test_label = np.append(test_label,
                                   np.array([idx for _ in range(self.__test_set_num)]).reshape(-1, 1),
                                   axis = 0)
                
        return train_data[1:], test_data[1:], train_label[1:], test_label[1:]
    
    #将一维数组分割，产生二维数组
    def __data_segmentation(self, arr, is_train = True):
        
        if is_train:
            new_arr = np.zeros((1, self.__sample_size))
            for i in range(self.__train_set_num):
                temp = arr[i * self.__data_augment_stride:\
                           i * self.__data_augment_stride +\
                           self.__sample_size].reshape(1, -1)
                new_arr = np.append(new_arr, 
                                    temp,
                                    axis = 0)
            return new_arr[1: ]
        
        else:
            new_arr = np.zeros((1, self.__sample_size))
            for i in range(self.__test_set_num):
                temp = arr[i * self.__sample_size:\
                           (i + 1) * self.__sample_size].reshape(1, -1)
                new_arr = np.append(new_arr, 
                                    temp,
                                    axis = 0)
            return new_arr[1:]

if __name__ == '__main__':
    
    reader = DataReader()
    x_train, x_test, y_train, y_test = reader.read_data(r'.\data\A')
        
