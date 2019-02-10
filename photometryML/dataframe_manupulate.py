#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:17:28 2018

@author: mirkazemi 
"""

import pandas as pd
import copy

def dataframe_list_statistics(df_list):
    '''
    For a list of pandas.DataFrames with similar indexing and columns name computes
    mean and STD at each cell. Dimensions of DataFrames should be equal.
    
    Parameters:
    -----------
    A list of pandas.DataFrames
    
    return:
    -------
    df_mean: A dataFrame with the same dimension of input DataFrames. Each cell
    represents the mean of similar cells in the input DataFrames.
    
    df_std: A dataFrame with the same dimension of input DataFrames. Each cell
    represents the standard deviation of similar cells in the input DataFrames.
    '''
    df_concat = pd.concat(df_list)
    by_row_index = df_concat.groupby(df_concat.index)
    df_mean = by_row_index.mean()
    df_std = by_row_index.std()
    return df_mean, df_std


def group_dataframe_list(input_df_list, column_name, column_values):
    '''
    For a list of pandas.DataFrames with a common column and list of values for
    that column provides a list DataFrames that each new DataFrames are made by 
    combination of rows with the same given value in target column.
    
    Parameters:
    -----------
    input_df_list: A list of pandas.DataFrames
    
    column_name: a column name available in all DataFrames in 'input_df_list'
    
    column_values: list of values for 'column_name'.
    
    return:
    -------
    output_df_list: A list of pandas.DataFrames. The size of list is equal to 
    size of 'column_name' and 'column_values'
    '''

    output_df_list = []
    for val in column_values:
        _dfs = [df[df[column_name] == val] for df in input_df_list]
        DF = pd.concat(_dfs)
        DF.reset_index(drop = True)
        output_df_list.append(DF)
    
    return output_df_list 

def combine_mean_STD_dfs(mean_df, STD_df, target_columns):
    '''
    Combines two 'mean' and 'STD' pandas.DataFrames, like output of 
    'dataframe_list_statistics'. It includes all column from 'mean_df' in 
    addition to 'target_columns' from 'STD_df'.
    
    parameters:
    -----------
    mean_df: a DataFrame containing mean values. All of its columns will be
    given in retrun.
    
    STD_df: a DataFrame containing standard deviation values. It's columns with
    name listed in 'target_columns' are included in return. An string '_err' 
    will be also added to the name of its column in returned dataFrame.
    
    return:
    -------
    A DataFrame 
    
    '''
    DF = copy.deepcopy(mean_df)
    init_cols = mean_df.columns.values.tolist()
    for col in target_columns:
        DF[col+"_err"] = STD_df[col]
        mean_index = init_cols.index(col)
        init_cols.insert(mean_index+1, col+"_err")
        
    return DF[init_cols]
    