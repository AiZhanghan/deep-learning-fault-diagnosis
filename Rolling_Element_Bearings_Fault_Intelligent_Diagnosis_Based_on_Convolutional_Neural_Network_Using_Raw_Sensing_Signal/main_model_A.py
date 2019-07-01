# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:22:09 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf
from DataReader import DataReader
from Preprocessor import Preprocessor

def minibatches(inputs = None, targets = None, batch_size = None, shuffle = False):
        
        assert len(inputs) == len(targets)
        
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

if __name__ == '__main__':
    
    path = '.\data\A'
    
    data_reader = DataReader()
    x_train, x_test, y_train, y_test = data_reader.read_data(path)
    
    preprocessor = Preprocessor()
    x_train, x_test = preprocessor.preprocess(x_train, x_test)
    