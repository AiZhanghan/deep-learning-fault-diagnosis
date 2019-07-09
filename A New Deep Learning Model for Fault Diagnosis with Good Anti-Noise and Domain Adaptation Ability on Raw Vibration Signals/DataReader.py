# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:52:05 2019

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
                 train_set_num = 660,
                 valid_set_num = 7):
        
        self.__data_augment_stride = data_augment_stride
        self.__sample_size = sample_size
        self.__test_set_num = test_set_num
        self.__train_set_num = train_set_num
        self.__valid_set_num = valid_set_num
    
    def read_data(self, path, sensor_place = ['DE', 'FE']):
        #所需数据长度，从最后一个开始，往回追溯
        length = ((self.__test_set_num + self.__valid_set_num) * self.__sample_size + 
                  self.__sample_size + (self.__train_set_num - 1) * self.__data_augment_stride)
        
        cate = [path + '\\' + x for x in os.listdir(path) if os.path.isdir(path + '\\' + x)]
        
        train_data = np.zeros((1, self.__sample_size, len(sensor_place)))
        test_data = np.zeros((1, self.__sample_size, len(sensor_place)))
        valid_data = np.zeros((1, self.__sample_size, len(sensor_place)))
        train_label = np.zeros((1))
        test_label = np.zeros((1))
        valid_label = np.zeros((1))
        
        for idx, folder in enumerate(cate):
            #读某一类数据,一类数据仅一个文件
            for t in glob.glob(folder + '\*.mat'):
                print('reading the data:%s'%(t))
                mat = loadmat(t)
                
                key = []
                for i in mat.keys():
                    for j in sensor_place:
                        if j in i:
                            key.append(i)
                            
                data_temp = np.append(mat[key[0]][-length:], mat[key[1]][-length:], axis = 1)
            
            train_data_length = (self.__sample_size +
                                 (self.__train_set_num - 1) * self.__data_augment_stride)
            
            train_data = np.append(train_data, 
                                   self.__data_segmentation(data_temp[: train_data_length]),
                                   axis = 0)
            
            test_data_length = self.__sample_size * self.__test_set_num
            
            test_data = np.append(test_data,
                                  self.__data_segmentation(
                                          data_temp[train_data_length: 
                                              train_data_length + test_data_length], 
                                              'test'),
                                    axis = 0)
                                      
            valid_data = np.append(valid_data, 
                                   self.__data_segmentation(
                                           data_temp[train_data_length + test_data_length:], 
                                           'valid'),
                                    axis = 0)
                                   
            train_label = np.append(train_label, 
                                    np.array([idx for _ in range(self.__train_set_num)]))
            
            test_label = np.append(test_label,
                                   np.array([idx for _ in range(self.__test_set_num)]))
            
            valid_label = np.append(valid_label,
                                    np.array([idx for _ in range(self.__valid_set_num)]))
                
        return(train_data[1:], 
               test_data[1:], 
               valid_data[1:],
               train_label[1:].astype(int), 
               test_label[1:].astype(int),
               valid_label[1:].astype(int))
        
        #将一维数组分割，产生二维数组
    def __data_segmentation(self, arr, set_type = 'train'):
        
        if set_type == 'train':
            new_arr = np.zeros((1, self.__sample_size, 2))
            for i in range(self.__train_set_num):
                temp = arr[i * self.__data_augment_stride:\
                           i * self.__data_augment_stride +\
                           self.__sample_size]
                new_arr = np.append(new_arr, 
                                    temp[np.newaxis, :],
                                    axis = 0)
            return new_arr[1: ]
        
        else:
            new_arr = np.zeros((1, self.__sample_size, 2))
            
            assert set_type == 'test' or set_type == 'valid'
            
            if set_type == 'test':
                set_num = self.__test_set_num
            else:
                set_num = self.__valid_set_num
                
            for i in range(set_num):
                temp = arr[i * self.__sample_size:\
                           (i + 1) * self.__sample_size]
                new_arr = np.append(new_arr, 
                                    temp[np.newaxis, :],
                                    axis = 0)
            return new_arr[1:]
        
if __name__ == '__main__':
    
    reader = DataReader(data_augment_stride=2048, train_set_num=3)
    x_train, x_test, x_valid, y_train, y_test, y_valid = reader.read_data(r'.\data\A')
