#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 21:25:41 2018

@author: mirkazemi
"""
import pandas as pd
import numpy as np
from . import machinelearning
from sklearn.model_selection import StratifiedKFold
from .dataframe_manupulate import dataframe_list_statistics
from .dataframe_manupulate import group_dataframe_list
from .dataframe_manupulate import combine_mean_STD_dfs
import itertools
import copy 

class Tuner():
    """
    Tuning a given hyperparameter of a classifier.
    
    parameters:
    -----------
    clf : a sklearn classifier object or a 'MLPipeline' object
    
    fixed_args: A dictionary of the parameters (arguments) of 'clf' which 
    have to be fixed
        
    variable_arg : A parameter name of 'clf' that is supposed to varie for 
    tuning
        
    variable_arg_values: Given values for chaning 'variable_param'
    
    n_splits : number of random validation and training sampling 
    std_scale : If True standardization scaling is applied.
    
    
    Methods
    -------
    tune : gets data and run tuning. The results are saved in 
    'predict_performance_mean' and 'predict_performance_std' attributes.
    
    Attributes:
    -----------
    classes : list of labels in 'y_data'
        
    accuracy : A dataframe including the means and standard deviations of
    classification accuracy for given range of 'variable_arg_values', for both
    train and validation samples.
    
    training_performance_list : list of performance dataframes for training 
    subsamples
    
    
    
        
    """
    def __init__(self, clf, fixed_args, variable_args, variable_args_values, 
                 n_splits = 4, std_scale = True, if_pca = False, pca_number = 2,
                 clf_type = 'clf'):
        self.clf = clf
        self.fixed_args = fixed_args
        self.variable_args = variable_args
        self.variable_args_values = variable_args_values
        self.n_splits = n_splits
        self.std_scale = std_scale
        self.clf_type = clf_type
        self.if_pca = if_pca
        self.pca_number = pca_number
        t = [ list(x) for x in itertools.product(*variable_args_values) ] 

        self.variable_arg_DF = pd.DataFrame(t,  columns = variable_args)
#        print(self.variable_arg_DF)

    def tune(self, data_x, data_y):
        '''
        Get X and Y data and evaluate the quality of classification
        '''
        SKF = StratifiedKFold(n_splits =self.n_splits, 
                              random_state = 1, 
                              shuffle=False)
        fit_accuracy_std = []
        fit_accuracy_mean = []
        predict_accuracy_std = []
        predict_accuracy_mean = []
        fit_performance_DFs = []
        predict_performance_DFs = []
        _args = copy.deepcopy(self.fixed_args)        
        self.classes = list(set(data_y))
        for i, row in self.variable_arg_DF.iterrows():
            _args.update(dict(zip(self.variable_args,row.values)))
#            print(_args)

#            print('self.fixed_args', self.fixed_args)
#            print('type:', type(self.clf))
            if self.clf_type == 'clf':
                pipe_ = machinelearning.MLPipeline(classifier = self.clf(**_args), 
                                                   classifier_name ='clf', 
                                       std_scale = self.std_scale,
                                       if_pca = self.if_pca,
                                       pca_number = self.pca_number)
                
            if self.clf_type == 'vote':
                pipe_ = machinelearning.MajorVoteClassifier(**self.fixed_args)
            
            # loop for k-fold cross validation
            _fit_accuracy_list = []
            _predict_accuracy_list = []
            _fit_performance_list = []
            _predict_performance_list = []
            
            for train_index, test_index in SKF.split(data_x, data_y):
                pipe_.fit(data_x.iloc[train_index,:], data_y.iloc[train_index])
                # prediction for validation subsample
                pipe_.predict(data_x.iloc[test_index,:], data_y.iloc[test_index])
#                print(pipe_.predict_accuracy)
                # storing accuracies and performances for any subsampling
                _fit_accuracy_list.append(pipe_.fit_accuracy)
                _predict_accuracy_list.append(pipe_.predict_accuracy)
                _fit_performance_list.append(pipe_.fit_performance)
                _predict_performance_list.append(pipe_.predict_performance)
            
            '''
            averaging over accuracies and performances for all 'val' in 'self.variable_args_values'
            '''
            # accuracies
            fit_accuracy_mean.append(np.mean(_fit_accuracy_list))
            fit_accuracy_std.append(np.std(_fit_accuracy_list))            
            predict_accuracy_mean.append(np.mean(_predict_accuracy_list))
            predict_accuracy_std.append(np.std(_predict_accuracy_list))
            
            # fitting performance     
            _df_mean, _df_STD = dataframe_list_statistics(_fit_performance_list)  
            _df_mean['class'] = self.classes
            _df_STD['class'] = self.classes
                
            _combined = combine_mean_STD_dfs(mean_df = _df_mean,
                                             STD_df = _df_STD, 
                                             target_columns = ['true_number','predicted_number','completeness','purity'])
#            _combined[self.variable_arg] = val
            fit_performance_DFs.append(_combined) 

            # prediction performance                 
            _df_mean, _df_STD = dataframe_list_statistics(_predict_performance_list)  
            _df_mean['class'] = pipe_.classes
            _df_STD['class'] = pipe_.classes
            _combined = combine_mean_STD_dfs(mean_df = _df_mean,
                                             STD_df = _df_STD, 
                                             target_columns = ['true_number','predicted_number','completeness','purity'])
#            _combined[self.variable_arg] = val          
            predict_performance_DFs.append(_combined)   
            
        '''
        summerzing the results 
        '''
        # accuracy:
        '''
        print('-----')
        print(self.variable_args, self.variable_args_values)
        print('-----')
        print('train_accuracy_mean', fit_accuracy_mean)
        print('train_accuracy_STD', fit_accuracy_std)
        print('validation_accuracy_mean', predict_accuracy_mean)
        print('validation_accuracy_STD', predict_accuracy_std)
        '''
        self.accuracy = self.variable_arg_DF 
        self.accuracy['train_accuracy_mean'] = fit_accuracy_mean
        self.accuracy['train_accuracy_STD'] = fit_accuracy_std
        self.accuracy['validation_accuracy_mean'] = predict_accuracy_mean
        self.accuracy['validation_accuracy_STD'] = predict_accuracy_std
    
        # performances       
        self.training_performance_list = group_dataframe_list(input_df_list = fit_performance_DFs,
                                                    column_name = 'class',
                                                    column_values = self.classes)
        
        self.validation_performance_list = group_dataframe_list(input_df_list = predict_performance_DFs,
                                                    column_name = 'class',
                                                    column_values = self.classes)        

            
        
      
       
