#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 18:15:36 2018

@author: mirkazemi
"""

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def performance(y_true, y_pred):
    """
    A function to evaluate the performance of a classifier by computing the
    'Completeness' and 'purity' of a claasification. 
    
    Parameters
    ----------
    'y_true' : array-like 
        Actual labels of the sample
    'y_pred' : array-like
        predicted labels of the sample from classification
    
    Returns
    -------
    _performance : pandas.DataFrame    
    The function returns a pandas.DataFrame object. Each row belongs to a label
    in the true labels list (unique values from 'y_true'). The columns are:
        1) 'label' 
        2) 'true_number': number of occurrence of a 'lebel' value in 'y_true'
        3) 'predicted_number': number of occurrence of a 'lebel' value in 'y_pred'
        4) 'completeness': fraction of objects with a given 'label' in 'y_true'  
            which are correctly predicted 
        5) 'purity': fraction of objects with given 'label' in 'y_pred' which
            are correctly predicted 
            
        Example for 'completeness' and 'putirty':
        We have a sample 100 email in which there are 40 spams (label 1). Our 
        classifier labels 35 of them as spam. 5 email of 35 classified emails 
        as spam are actually not spams. Thus the purity for label 1 (spams) is 
        30/35. 10 of 40 actual spams are classified as normal email (label 0). 
        Thus we lost 10 of spams. Thus the completness is (40-10)/40.
        confusion matrix:
        [55    5]
        [10   30]        
        
    """
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    _performance = pd.DataFrame({'class': list(set(y_true))})
    _performance['true_number'] = np.sum(_confusion_matrix, axis=1)
    _performance['predicted_number'] = np.sum(_confusion_matrix, axis=0)
    _performance['completeness'] = _confusion_matrix.diagonal()/_performance['true_number']
    _performance['purity'] = _confusion_matrix.diagonal()/_performance['predicted_number']
    return _performance

def score(y_true, y_pred, target_param = 'purity', target_label = 0):
    """ 
    a fucntion is written as an score evaluating for scoring during fine tuning
    procedures base on performance() function. It computes the performance 
    DataFrame by calling performance() function and returns the targeted parameter
    for a given label.
    Parameters
    ----------
    'y_true' : array-like 
        Actual labels of the sample
        
    'y_pred' : array-like 
        Predicted labels of the sample from classification
        
    'target_label' : int, string, float
        The label for which the user requests
        
    'target_parameter' : string
        The parameter that is used for scoring. 'purity' or 'completeness'. 
        by default it is set to 'purity'
    """
    _performance = performance(y_true, y_pred)
    return _performance.loc[_performance['class'] == target_label, target_param].values[0]



class ROC():
    """
    Compute Receiver operating characteristic (ROC) for a group of classifiers 
    using stratified k-fold cross-validation approach. 
    It uses 'sklearn.metrics.roc_curve' function and 
    'sklearn.model_selection.StratifiedKFold' class.
    
    Parameters
    ----------
    classifiers : list of clasiifiers
    
    classifiers_name : list of classifiers name
    
    data_x : a pandas.DataFrame or array of features (n_samples, n_features)
    
    data_y : an array-like, shape (n_samples,). The predicted classes.
    
    pos_label : int or str, default=None
                Label considered as positive and others are considered negative.
                
    n_splits : Number of folds. Must be at least 2.
    
    random_state : int, RandomState instance or None, optional, default=None
    
    Methods and Attributes
    ----------------------
    roc_curve : A list of pandas DataFrame. The order of DataFrames are similar 
        to given 'classifiers'  ROC curve with three columns:
        1) 'FP_rate_mean' : Flase Positive rates, X-axis of ROC
        2) 'TP_rate_mean' : True Positive rates, Y-axis of ROC
        3) 'TP_rate_err'  : Error of True Positive rates, Y-axis error in ROC
    
    
        
    """
    def __init__(self,classifiers, classifiers_name, data_x, data_y, pos_label, n_splits=3, random_state=None):
        self.classifiers_name = classifiers_name
        FP_rate_mean = np.linspace(0, 1, 100) # X-axis in ROC
        SKF = StratifiedKFold(n_splits = n_splits, 
                              random_state = random_state, 
                              shuffle=False)
        """
        if isinstance(data_x, pd.DataFrame):
            print('yessss')
            data_x = data_x.values
        if isinstance(data_y, pd.DataFrame):     
            data_y = data_y.values
        """
        self.roc_curve = []
        self.AUCs = []
        self.AUCs_err = []
        i = 0
        for i, _classifier in enumerate(classifiers):
            
            _TP_rates = []
            _AUCs = []
            for train_index, test_index in SKF.split(data_x, data_y):
                # train on train sample:
                _classifier.fit(data_x.iloc[train_index,:], data_y.iloc[train_index])
        
                # index of posotive label 'pos_label' in classifeir.class_
                _pos_label_index = np.where(_classifier.classes_ == pos_label)[0][0]
            
                # computing the probability for test sample
                probas = _classifier.predict_proba(data_x.iloc[test_index,:])
                FP_rate, TP_rate, thresholds = roc_curve(y_true = data_y.iloc[test_index],
                                                         y_score = probas[:, _pos_label_index],
                                                         pos_label = pos_label)    
        
        
                _TP_rates.append(interp(x = FP_rate_mean, xp = FP_rate, fp = TP_rate))
                _TP_rates[-1][0] = 0.0
            
                _AUCs.append(auc(FP_rate, TP_rate))

            _roc_curve = pd.DataFrame({'FP_rate_mean' : FP_rate_mean,
                                       'TP_rate_mean' : np.mean(_TP_rates, axis = 0),
                                       'TP_rate_err' : np.std(_TP_rates, axis = 0)})
            _roc_curve[-1,'TP_rate_mean'] = 1.0
        
            self.roc_curve.append(_roc_curve)
            self.AUCs.append(np.mean(_AUCs))
            self.AUCs_err.append(np.std(_AUCs))
        
            
    def plot(self, color = ['b', 'r', 'g', 'm' , 'k'], 
             plot_label = 'Receiver operating characteristic', file = None):
        fig = plt.figure()
        plt.plot([0, 1],[0, 1],linestyle = '--',color = 'grey')
        for i, clf_name in enumerate(self.classifiers_name):
            plt.plot(self.roc_curve[i]['FP_rate_mean'], self.roc_curve[i]['TP_rate_mean'], 
                     color=color[i], label=r'%s ROC (AUC = %0.3f $\pm$ %0.3f)' % (self.classifiers_name[i], self.AUCs[i], self.AUCs_err[i] ), lw=2, alpha=.8)     
        
            TP_upper = np.minimum(self.roc_curve[i]['TP_rate_mean'] + self.roc_curve[i]['TP_rate_err'], 1)
            TP_lower = np.maximum(self.roc_curve[i]['TP_rate_mean'] - self.roc_curve[i]['TP_rate_err'], 0)        
            plt.fill_between(self.roc_curve[i]['FP_rate_mean'], TP_upper, TP_lower, color=color[i], alpha=.2) 
    
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(plot_label)
        plt.legend(loc="lower right")
        plt.show()
        if file is not None:
            fig.savefig(file)



'''

class ROC():
    """
    Compute Receiver operating characteristic (ROC) using stratified k-fold 
    cross-validation approach. It uses 'sklearn.metrics.roc_curve' function and
    'sklearn.model_selection.StratifiedKFold' class.
    
    Parameters
    ----------
    classifier : a clasiifier
    data_x : a pandas.DataFrame or array of features (n_samples, n_features)
    data_y : a array-like, shape (n_samples,). The predicted classes.
    pos_label : int or str, default=None
                Label considered as positive and others are considered negative.
    n_splits : Number of folds. Must be at least 2.
    random_state : int, RandomState instance or None, optional, default=None
    
    Returns
    -------
    a pandas.dataFrame : 
        
    """
    def __init__(self,classifier, data_x, data_y, pos_label, n_splits=3, random_state=None):
        FP_rate_mean = np.linspace(0, 1, 100) # X-axis in ROC
        SKF = StratifiedKFold(n_splits = n_splits, 
                              random_state = random_state, 
                              shuffle=False)
        """
        if isinstance(data_x, pd.DataFrame):
            print('yessss')
            data_x = data_x.values
        if isinstance(data_y, pd.DataFrame):     
            data_y = data_y.values
        """
        self.TP_rates = []
        self.AUCs = []
        i = 0
        for train_index, test_index in SKF.split(data_x, data_y):
            print(i)
            i += 1
            # train on train sample:
            classifier.fit(data_x.iloc[train_index,:], data_y.iloc[train_index])
        
            # index of posotive label 'pos_label' in classifeir.class_
            _pos_label_index = np.where(classifier.classes_ == pos_label)[0][0]
            
            # computing the probability for test sample
            probas = classifier.predict_proba(data_x.iloc[test_index,:])
            FP_rate, TP_rate, thresholds = roc_curve(y_true = data_y.iloc[test_index],
                                                     y_score = probas[:, _pos_label_index],
                                                     pos_label = pos_label)    
        
        
            self.TP_rates.append(interp(x = FP_rate_mean, xp = FP_rate, fp = TP_rate))
            self.TP_rates[-1][0] = 0.0
            
            self.AUCs.append(auc(FP_rate, TP_rate))
#        print(self.TP_rates)
        self.roc_curve = pd.DataFrame({'FP_rate_mean' : FP_rate_mean,
                                       'TP_rate_mean' : np.mean(self.TP_rates, axis = 0),
                                       'TP_rate_err' : np.std(self.TP_rates, axis = 0)})
        print(self.AUCs)
        self.roc_curve[-1,'TP_rate_mean'] = 1.0
            
            
    def plot(self,file = None):
        plt.plot([0, 1],[0, 1],linestyle = '--',color = 'r')
        plt.plot(self.roc_curve['FP_rate_mean'], self.roc_curve['TP_rate_mean'], 
                 color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (np.mean(self.AUCs), np.std(self.AUCs )), lw=2, alpha=.8)     
        
        TP_upper = np.minimum(self.roc_curve['TP_rate_mean'] + self.roc_curve['TP_rate_err'], 1)
        TP_lower = np.maximum(self.roc_curve['TP_rate_mean'] - self.roc_curve['TP_rate_err'], 0)
            
        plt.fill_between(self.roc_curve['FP_rate_mean'], TP_upper, TP_lower, color='b', alpha=.2,
                         label=r'$\pm$ 1 std. dev.') 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig('foo.png')
        
'''
        
        