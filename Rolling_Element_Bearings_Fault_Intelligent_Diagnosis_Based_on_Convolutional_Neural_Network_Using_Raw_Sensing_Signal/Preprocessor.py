# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:12:17 2019

@author: Administrator
"""

from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    
    def preprocess(self, x_train, x_test, x_valid):
        
        min_max_scaler = MinMaxScaler()

        x_train= min_max_scaler.fit_transform(x_train)
        
        x_test = min_max_scaler.transform(x_test)
        
        x_valid = min_max_scaler.transform(x_valid)
    
        return x_train, x_test, x_valid
    