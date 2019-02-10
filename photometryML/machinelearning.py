#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:34:05 2018

@author: mirkazemi
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .performance import performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from .dataframe_manupulate import dataframe_list_statistics
from .dataframe_manupulate import group_dataframe_list
from .dataframe_manupulate import combine_mean_STD_dfs
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from scipy import interp

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

class MLPipeline(Pipeline):
    """
    ML_pipeline: Machine learning pipeline
    It is a modified version of sklearn.pipeline.Pipeline.
    It has all the functionalities of sklearn.pipeline.Pipeline in addition to 
    saving the performance of the fitting ('fit' method) and prediction 
    ('predict' method). The __init__ funtion is also modified in a way that the 
    user do not separately mentioned the PCA and standardization tool in a an 
    input list. The user only need to create one the classifiers in 'sklearn' 
    package. 
    
    It has following additional attributes:
        
        1) ML_pipeline.fit_confusion_matrix (type: matrix)
        2) ML_pipeline.fit_accuracy (numeric)
        3) ML_pipeline.fit_performance (pandas.DataFrame)
        Confusion matrix, accuracy and peformance of the fitting for train 
        sample. Whenever the classifier is trained by calling '.fit' method on 
        a training sample, they are automatically computed. Computing these 
        three attributes are the only differences between 'ML_pipeline.fit' and
        'sklearn.pipeline.Pipeline.fit'. All other functionallities are the 
        same.
        
        4) ML_pipeline.predict_confusion_matrix (type: matrix)
        5) ML_pipeline.predict_accuracy (numeric)
        6) ML_pipeline.predict_performance (pandas.DataFrame)        
        Similar to 1,2 and 3 but they are calculated when 'predict' function is 
        called. Attention: as mentioned before all the methods in 
        'sklearn.pipeline.Pipeline' are available in 'ML_pipeline' too but
        4, 5, and 6 are only calculated when 'predict' method is called and not
        'predict_proba' or 'predict_log_proba' are called.
        
        1, 2, 3, 4, 5 and 6 are initiated by None. Their value are not None only
        after calling 'fit' or 'predcit' methods.
    
    Parameters
    ----------
    'classifier': an object from classifier classes in 'sklearn' package. 
    For example: sklearn.neural_network.MLPClassifier or sklearn.svm.SVC objects.
    
    'std_scale': If True, sklearn.preprocessing.StandardScaler is included in the
    pipeline. Default: True
    
    'pca': If True, 'principal component analysis' (PCA) is applied by 
    including a sklearn.decomposition.PCA object in the pipeline to extract 
    'n_components' number of features. The goal is to reduce the dimensionality.
    Default: False
    
    'n_components': number of extracted features in PCA method. The same argument
    as in 'sklearn.decomposition.PCA'.
        
    Example:
    from sklearn.svm import SVC
    
    my_classifier = SVC(kernel='linear', C=0.1,random_state=0, 
                        class_weight = 'balanced', probability = True)
    
    my_pipeline = ML_pipeline(classifier = my_classifier, std_scale = True)
    
    The above object 'my_pipeline' is pipeline with standardization and SVC
    classifier.

    """
    def __init__(self, classifier, classifier_name, std_scale = False, if_pca = False, pca_number = 2):
        self.classifier = classifier
        steps = []
        if std_scale:
            steps.append(('sc',StandardScaler()))
            
        if if_pca:
            steps.append(('pca', PCA(n_components = pca_number)))
        steps.append((classifier_name, self.classifier))
        
        super(MLPipeline,self).__init__(steps)
        
        self.fit_confusion_matrix = None
        self.fit_accuracy = None
        self.fit_performance = None
        self.predict_confusion_matrix = None
        self.predict_accuracy = None
        self.predict_performance = None
        
    def fit(self,data_x, data_y):
        if isinstance(data_x, pd.DataFrame):
            self.x_data_colnames = list(data_x.columns.values)
        else:
            self.x_data_colnames = []
        
        self.classes = list(set(data_y))
        super(MLPipeline,self).fit(data_x, data_y)
        y_pred =  self.predict(data_x)
        self.fit_confusion_matrix = confusion_matrix(data_y, y_pred)
        self.fit_accuracy = accuracy_score( y_true = data_y, y_pred = y_pred )
        self.fit_performance = performance( y_true = data_y, y_pred = y_pred )
        del y_pred
        del data_x
        del data_y        
        
    def fit_with_error(self, data_x, data_x_err, data_y, iter_num = 10):
        self.classes = list(set(data_y))

        '''
        Saving the column name if the incput data_x is a DataFrame and covert 
        it to numpy array.
        '''        
        if isinstance(data_x, pd.DataFrame):
            self.x_data_colnames = list(data_x.columns.values)
            _data_x = copy.deepcopy(data_x.values)
        else:
            self.x_data_colnames = []
            _data_x = copy.deepcopy(data_x)
                
        '''
        generating noisy data and added it to input train sample:
        '''
        
        for i in range(1,iter_num):
            _data_x_additioal = np.random.normal(data_x, data_x_err)

            _data_x = np.vstack((_data_x, _data_x_additioal))
        _data_y = np.tile(data_y, iter_num)
        
        '''
        training:
        '''
        super(MLPipeline,self).fit(_data_x, _data_y)        
        
        '''
        performance evaluation:
        '''
        y_pred =  self.predict(data_x)
        self.fit_confusion_matrix = confusion_matrix(data_y, y_pred)
        self.fit_accuracy = accuracy_score( y_true = data_y, y_pred = y_pred )
        self.fit_performance = performance( y_true = data_y, y_pred = y_pred )
        del y_pred
        del data_x
        del data_y            



    def predict(self,data_x, data_y = None):   
        
        '''
        If self.x_data_colnames != [], it means that the MLPipeline was fitted
        by a pandas.DataFrame as data_x. Thus the order of data_x for prediction
        should be the same as the data_x for fitting. If self.x_data_colnames != [] 
        is False, it means that the fitting was done using a numpy.array and 
        the user should be aware of importance data_x orders (features order).
        Prediction:
        '''
        if self.x_data_colnames != [] and isinstance(data_x, pd.DataFrame):
            y_pred = super(MLPipeline,self).predict(X = data_x[self.x_data_colnames].values)
        else: 
            y_pred = super(MLPipeline,self).predict(X = data_x)
        
        '''
        If data_y is provided, the performance of the prediction will be
        evaluated and stored:
        '''
        if data_y is None:
            self.predict_confusion_matrix = None
            self.predict_accuracy = None
            self.predict_performance = None
        else:            
            self.predict_confusion_matrix = confusion_matrix(data_y, y_pred)
            self.predict_accuracy = accuracy_score( y_true = data_y, y_pred = y_pred )
            self.predict_performance = performance( y_true = data_y, y_pred = y_pred )
            
        del data_y             
        del data_x
        return y_pred

    def predict_proba_with_error(self, data_x, data_x_err, iter_num = 10):   
    
        '''
        If self.x_data_colnames != [], it means that the MLPipeline was fitted
        by a pandas.DataFrame as data_x. Thus the order of data_x for prediction
        should be the same as the data_x for fitting. If self.x_data_colnames != [] 
        is False, it means that the fitting was done using a numpy.array and 
        the user should be aware of importance data_x orders (features order).
        Probability prediction:
        '''        
        if self.x_data_colnames != [] and isinstance(data_x, pd.DataFrame):
            y_proba = super(MLPipeline,self).predict_proba(X = data_x[self.x_data_colnames])
        else: 
            y_proba = super(MLPipeline,self).predict_proba(X = data_x)
                
        '''
        In the following loop, noisy data_x samples are generated and the 
        probabilites are computed for them. The probability error is the 
        standard deviation of the probabilities of simulated data_xs: 
        '''
        _sim_y_proba_list = []

        for i in range(0,iter_num):
            _data_x = np.random.normal(data_x, data_x_err) 
            _sim_y_proba_list.append(super(MLPipeline,self).predict_proba(X = _data_x))
        
        y_proba_err = np.std(_sim_y_proba_list, axis = 0)
        
        return y_proba, y_proba_err


    def learning_curve(self, data_x, data_y, 
                       train_size_steps = np.linspace(0.1, 1.0, 10),
                       iter_num = 4,
                       valid_frac = 0.2,
                       pos_label = 0):
        
        _valid_flag = np.random.random(len(data_x)) < valid_frac
        valid_x = copy.deepcopy(data_x[_valid_flag])
        valid_y = copy.deepcopy(data_y[_valid_flag])
        
        init_train_x = copy.deepcopy(data_x[~_valid_flag])
        init_train_y = copy.deepcopy(data_y[~_valid_flag])
        
        fit_performance_DFs = []
        predict_performance_DFs = []       
        
        self.learning_curve_fit = []
        self.learning_curve_predict = []
        
        # itteration for fitting on 'train_size_steps' fraction of train data:
        for i, _size_step in enumerate(train_size_steps):
            
            _fit_DF_list = []
            _predict_DF_list = []
            
            # k-fold for subsamples for fitting   
            if _size_step != 1.0:
                for j in range(iter_num):
                    _flag = np.random.random(len(init_train_x)) < _size_step                    
                    self.fit(init_train_x[_flag], init_train_y[_flag])
                    self.predict(valid_x, valid_y)
                    _fit_DF_list.append(self.fit_performance)
                    _predict_DF_list.append(self.predict_performance)
                    
            # one fitting on whole train sample
            else:
                self.fit(init_train_x, init_train_y)
                self.predict(valid_x, valid_y)
                _fit_DF_list.append(self.fit_performance)
                _predict_DF_list.append(self.predict_performance)               
            
            # computing mean and STD for each train sample and add to the list
            _df_mean, _df_STD = dataframe_list_statistics(_fit_DF_list)
            _df_mean['class'] = self.classes
            _df_STD['class'] = self.classes
            _combined = combine_mean_STD_dfs(mean_df = _df_mean,
                                             STD_df = _df_STD, 
                                             target_columns = ['true_number','predicted_number','completeness','purity'])
            _combined['train_frac'] =_size_step
            fit_performance_DFs.append(_combined)       
            
            # computing mean and STD for each validation sample and add to the list
            _df_mean, _df_STD = dataframe_list_statistics(_predict_DF_list)   
            _df_mean['class'] = self.classes
            _df_STD['class'] = self.classes            
            _combined = combine_mean_STD_dfs(mean_df = _df_mean,
                                             STD_df = _df_STD, 
                                             target_columns = ['true_number','predicted_number','completeness','purity'])
            _combined['train_frac'] =_size_step
            predict_performance_DFs.append(_combined)   
            
                        
        
        self.learning_curve_fit = group_dataframe_list(input_df_list = fit_performance_DFs,
                                                       column_name = 'class',
                                                       column_values = self.classes)            
        self.learning_curve_predict = group_dataframe_list(input_df_list = predict_performance_DFs,
                                                           column_name = 'class',
                                                           column_values = self.classes)


        

                                                              
        
        
class MajorVoteClassifier():
    """
    Majority vote ensemble classifier
    
    Parameters
    ----------
    classifiers: array-like, shape = [n_classifiers]
    list MLPipeline objects. Each MLPipeline belongs to one specific
    classifier.
    
    vote_type: str, {'label', 'probability'}
    
    clf_weights: array-like, shape = [n_classifiers]
    Default: None
    If None, all classifiers have the same weight in voting.
    """
    def __init__(self, classifiers_pipeline  = [], classifiers_name = [], vote_type = 'label', clf_weights = None):
        if len(classifiers_pipeline) != len(classifiers_name):
            print("Error: Length of 'classifiers_pipeline' does not match to the length of 'classifiers_name'")
            return -1
        if (clf_weights is not None) and (len(classifiers_pipeline) != len(clf_weights)):
            print("Error: Length of 'classifiers_pipeline' does not match to the length of 'clf_weights'")
            return -1
        self.classifiers_pipeline = dict(zip(classifiers_name, classifiers_pipeline))
        self.classifiers_name = classifiers_name
        self.vote_type = vote_type
        if clf_weights is None:
            clf_weights = np.repeat(0.4,len(classifiers_pipeline))
        self.clf_weights = np.array(clf_weights)/sum(clf_weights)
        
    def fit(self, data_x, data_y):
        [self.classifiers_pipeline[x].fit(data_x, data_y) for x in self.classifiers_name]
        self.classes = list(set(data_y))
#        print(self.classifiers_pipeline['svm'].fit_performance)
        y_pred =  self.predict(data_x)
        self.fit_confusion_matrix = confusion_matrix(data_y, y_pred)
        self.fit_accuracy = accuracy_score( y_true = data_y, y_pred = y_pred )
        self.fit_performance = performance( y_true = data_y, y_pred = y_pred )
        
    def predict(self, data_x, data_y = None):
        if self.vote_type == 'label':        
            self.y_pred_df = pd.DataFrame.from_items(zip(self.classifiers_name, 
                                                    [self.classifiers_pipeline[x].predict(data_x, data_y) for x in self.classifiers_name]))  

            _votes = np.array([np.sum( (self.y_pred_df[self.classifiers_name] == label) * self.clf_weights, axis = 1)  for label in self.classes])
            y_pred = np.array([self.classes[x] for x in np.argmax(_votes, axis = 0)])
            self.y_pred_df['major_vote'] = y_pred
            

        if self.vote_type == 'probability':
            
            # computing probabilities using 'predict_proba()'
            probas = self.predict_proba(data_x)
            
            # finding the class with maximum probability for each data point
            y_pred = np.array(self.classes)[np.argmax(probas,axis=1)]

        # perfomance evalutaion for the prediction if test data_y is provided
        if data_y is None:
            self.predict_confusion_matrix = None
            self.predict_accuracy = None
            self.predict_performance = None
        else:            
            self.predict_confusion_matrix = confusion_matrix(data_y, y_pred)
            self.predict_accuracy = accuracy_score( y_true = data_y, y_pred = y_pred )
            self.predict_performance = performance( y_true = data_y, y_pred = y_pred )
            del data_y 
            
        return y_pred
    
    def predict_proba(self, data_x):
        # getting the probabilities from each classifier in a lsit of array
        _classifiers_proba = [self.classifiers_pipeline[x].predict_proba(data_x) for x in self.classifiers_name]

        # multiplying the probabilities from each classifier by classifier weight
        _weighted_classifiers_proba = np.array([x*y for x,y in zip(self.clf_weights,_classifiers_proba)])
        del _classifiers_proba
            
        # summing up the probabilities from each classifier (the clf_weights have been already normalized)
        _summed_weighted_proba = np.sum(_weighted_classifiers_proba,axis=0)
        del _weighted_classifiers_proba
        return _summed_weighted_proba
    
    