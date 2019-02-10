#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:43:45 2018

@author: mirkazemi
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
     


def plot_decision_regions(X, x_columns, y_column, classifier, 
                          plot_title = None, xlabel = None, ylabel = None,
                          resolution=0.05, plotFile = None):

    """
    Plotting decision regions and data points. It only works for plotting the
    regions and data points for a machine learning model with two features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        A dataframe consists of features and label
        
    x_columns : list
        A list with length of 2. It includes column names of two features.
    
    y_column : string
        Name of column for labels.
        
    plot_title : string, optional
        Title of plot.
        
    xlabel : string
        Label for the X-axis of the plot.
        
    ylabel : string
        Label for the Y-axis of the plot.
        
    resolution : double
        The size of regions for sampling the prediction. default value set to 0.05
        
    plotFile : string, optional
        File name for saving the plot. The plot will save only if a name is
        given.
        
    """    
    # setup marker generator and color map
    fig = plt.figure(figsize=(9, 6))
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(X[y_column]))])
    cmap_countour = plt.cm.RdBu
       # plot the decision surface
       
    x0_min, x0_max = X[x_columns[0]].min() - 1, X[x_columns[0]].max() + 1
    x1_min, x1_max = X[x_columns[1]].min() - 1, X[x_columns[1]].max() + 1
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, resolution),
                           np.arange(x1_min, x1_max, resolution))
    try:
        Z = classifier.predict_proba(np.array([xx0.ravel(), xx1.ravel()]).T)
        Z = Z[:,1]
    except AttributeError:
        Z = classifier.predict(np.array([xx0.ravel(), xx1.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx0, xx1, Z, levels=[0,0.5,1], alpha=0.6, cmap=cmap_countour)
    plt.xlim(xx0.min(), xx0.max())
    plt.ylim(xx1.min(), xx1.max())
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(plot_title)

    _X = X.sample(frac=1)

    plt.scatter(x = _X[x_columns[0]], y=_X[x_columns[1]], s = 6,
                alpha = 0.3, c = cmap(_X[y_column]))       
    if plotFile is not None: 
        fig.savefig(plotFile)
    plt.show()    
    
    
