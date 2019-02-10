#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:09:52 2018

@author: mirkazemi
"""

from .performance import *
import numpy as np
label = [1,1,1,1,1,1,1,1,1,1,
         2,2,2,2,2,2,
         0,0,0,0,0,0]
label_pred = [1,0,2,2,1,1,1,1,0,0,
              2,2,2,2,0,0,
              0,0,0,1,2,2]

def test_performance():
    perf = performance(y_true = label,
                       y_pred = label_pred)
    np.testing.assert_array_almost_equal(perf['completeness'].values, [0.5, 0.5, 0.666667])
    np.testing.assert_array_almost_equal(perf['purity'].values, [0.375, 0.833333, 0.5])
    
def test_score():
    np.testing.assert_almost_equal(score(y_true = label, y_pred = label_pred, target_param = 'purity', target_label = 1), 0.8333333333)
    np.testing.assert_almost_equal(score(y_true = label, y_pred = label_pred, target_param = 'completeness', target_label = 0), 0.5)

