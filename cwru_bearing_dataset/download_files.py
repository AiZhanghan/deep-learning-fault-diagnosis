# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:05:31 2019

@author: Administrator
"""

import cwru

def download_files():
    
    exps =  ['12DriveEndFault', '12FanEndFault', '48DriveEndFault']
    rpms = ['1797', '1772', '1750', '1730']
    
    for exp in exps:
        for rpm in rpms:
            data = cwru.CWRU(exp, rpm, 384)

if __name__ == '__main__':
    
    download_files()