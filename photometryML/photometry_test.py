#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:01:44 2018

@author: mirkazemi
"""

from .photometry import *
import pandas as pd
import numpy as np 

data = pd.DataFrame({'groupID' : [0,1,1,2,3,3,4,5],
                     'RA' : [0, 1, 2, 3, 4, 5, 6, 7],
                     'Dec' : [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                     'R' : [19, 22, 23.4, -9999, 18.4, 20.1, -9999, 21.8],
                     'Rerr' : [0.01, 0.02, 0.04, -9999, 0.01, 0.02, -9999, 0.03],
                     'G' : [19.5, 22.9, 23.8, 23.6, 19.4, 20.7, 23.4, 22.5],
                     'Gerr' : [0.01, 0.02, 0.04, 0.05, 0.01, 0.02, 0.05, -9999],
                     'label' : [0,1,0,0,0,1,0,1]})

def test_PhotomCleanData():
    p1 = PhotomCleanData(input_data = data,  missing_value = -9999 )
    data1 = p1.return_data(init_x_columns = ['RA', 'Dec'],
                                       init_x_columns_err = [ None, None],
                                       mag_columns_4color1 = ['G'], 
                                       mag_columns_4color2 = ['R'],                   
                                       mag_columns_4color1_err = ['Gerr'], 
                                       mag_columns_4color2_err = ['Rerr'],
                                       y_column = 'label',
                                       valid_values_only = True)

    np.testing.assert_equal(data1.index.values, [0, 1, 2, 4, 5])
    
def test_data_partition_cc():
    train, valid, test = data_partition_cc(data = data, train_frac = 0.6, valid_frac = 0.0, test_frac = 0.4, 
                              grouping_column = None, random_seed = 1000, return_boolean = False)
   
    np.testing.assert_equal(test.index.values, [0, 2, 4])
    
    train, valid, test = data_partition_cc(data = data, train_frac = 0.6, valid_frac = 0.0, test_frac = 0.4, 
                              grouping_column = None, random_seed = 1500, return_boolean = True)
    
    np.testing.assert_equal(train, [False, True, False, True, False, False, True, True])  
    
    train, valid, test = data_partition_cc(data = data, train_frac = 0.6, valid_frac = 0.0, test_frac = 0.4, 
                              grouping_column = 'groupID', random_seed = 1000, return_boolean = True)

    np.testing.assert_equal(train, [False, True, True, False, True, True, False, True])



