#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:17:46 2018

@author: mirkazemi
"""

import numpy as np
import copy
import pandas as pd

"""
The module includes a class and a function:
    1) PhotomCleanData()
       A class which get a dataFrame of photmetric data and upon request returns
       specific columns, produces colors and their errors from magniude columns
       their error and excludes data points with missed values.
       
    2) data_partition_cc()
       A function that gets a DataFrame and parts into train, validation, and 
       test sample. Its advantage over 'sklearn.model_selection.train_test_split'
       is it can divide the sample base on a common group name.   
"""

class PhotomCleanData():
    
    """
    Photometric clean data producer:
    An objects made by this class saves a pandas.DataFrame. When the user requests
    for a set of columns in photometric dataset it returns a subsample of
    original dadaset. The subsample consists of rows that have no missing values in 
    their intended columns. It can also be asked to return a dataFrame including
    subtraction ofbetween some columns (useful for creating color columns from 
    magnitudes). 
     
    Attributes
    ----------
    input_data : numpy.array or pandas.DataFrame
        Input photometric data set.
        
    missing_value : float or int
        The value in the input data set 'input_data' showing that the data is 
        not available (the default is -99.0).
        
    Methods
    -------
    return_data(init_x_columns, init_x_columns_err, mag_columns_4color1 = [], 
             mag_columns_4color2, mag_columns_4color1_err, mag_columns_4color2_err,
             y_column = None, valid_values_only = True)
        Returns desired magnitudes, colors and their errors in addition other
        requested columns by removing rows with missing data.
 
    """

    def __init__(self, input_data,  missing_value = -99.0 ):
        """
        Parameters
        ----------
        input_data : input data in the format of pandas.DataFrame
        missing_value : a value showing that a cell in data frame has been missed.
            default: -99.0

        Please see the example in 'get_data' method.
        """
        self.input_data = input_data
        self.missing_value = missing_value
    
    def return_data(self, init_x_columns = [],
                 init_x_columns_err = [],
                 mag_columns_4color1 = [], 
                 mag_columns_4color2 = [],                   
                 mag_columns_4color1_err = [], 
                 mag_columns_4color2_err = [],
                 y_column = None,
                 valid_values_only = True):
        """    
        Parameters
        ----------     
        init_x_columns : list
            List of columns name for features excluding colors that are expected
            to be created by this class
            
        init_x_columns_err : list, optional
            List of columns name representing the errors for 'init_x_columns'. 
            The order and size of 'init_x_columns' and 'init_x_columns_err' must
            be similar.
            
        mag_columns_4color1 : list, optional 
            List of magnitudes names (columns name) that are used as the first 
            magnitude in the color computation (m1 in c = m1 - m2).
        
        mag_columns_4color2 : list, optional
            List of magnitudes names (columns name) that are used as the second 
            magnitude in the color computation (m2 in c = m1 - m2).
        
        mag_columns_4color1_err : list, optional
            List of columns name representing the errors for 'mag_columns_4color1'. 
            The order and size of 'mag_columns_4color1' and 'mag_columns_4color1_err' must
            be similar.
        
        mag_columns_4color2_err : list, optional
            List of columns name representing the errors for 'mag_columns_4color2'. 
            The order and size of 'mag_columns_4color2' and 'mag_columns_4color2_err' must
            be similar.        
            
        y_column: string, optional
            Y column name 
    
        Returns
        -------
        A pandas.DataFrame including:
            1) columns that their name are listed in 'x_columns'
            2) new columns which are created by subtracting features listed in 
               'mag_columns_4color1' by features listed in 'mag_columns_4color2'
            3) column that its name is given by "y_column" if "y_column" is given.
        
        
        Assume tht the follwoing in the original dataset:
    
            init_data:
    
                    G     R     I     type
                0   22.5  21.4  21.0  "A7"
                1   19.2  18.9  18.8  "A4"
                2   17.1  -99.0 17.8  "B2"
                3   23.3  23.8  24.1  "A7"
    
        Creating an object:
            > data_prod = photom_data_producer(init_data=init_data, missing_value = -99.0)
    
        > data = data_prod.get_data(x_columns = ['G', 'R'],
                                    mag_columns_4color1 = ['R'], 
                                    mag_columns_4color2 = ['I'],
                                    y_column = 'type')
    
        > print(data)
            G     R     R-I     type
        0   22.5  21.4  0.4     "A7"
        1   19.2  18.9  0.1     "A4"
        3   23.3  23.8  -0.3    "A7"    
    
        The row '2' is not returned because 'R' value is missed.

        Hint: The new columns that are created by mag_columns_4color1' and
        'mag_columns_4color2' are included in the 'photom_data_producer.x_columns'
        attribure. So:
        
            > print(data_prod.x_columns)
            ['G', 'R', 'R-I']
            > print(data_prod.y_column)
            'type'
            """
    
        # check if the argument lengths are correct
        
        if (len(init_x_columns) != len(init_x_columns_err) and init_x_columns_err!=[]):
            print('ERROR: length of "x_columns" and "x_columns_err" are not equal')      
            
        if (len(mag_columns_4color1) != len(mag_columns_4color2)):
            print('ERROR: length of "mag_columns_4color1" and "mag_columns_4color1" are not equal')
            
        if (len(mag_columns_4color1_err) != len(mag_columns_4color1) and mag_columns_4color1_err != []):
            print('ERROR: length of "mag_columns_4color1_err" and "mag_columns_4color1" are not equal')                
        
        if (len(mag_columns_4color1_err) != len(mag_columns_4color2_err) and mag_columns_4color1_err != []):
            print('ERROR: length of "mag_columns_4color1_err" and "mag_columns_4color2_err" are not equal')        

        '''
        Saving important input as class attributes. 'self.x_columns' and 
        'self.x_columns_err' are initiated by 'init_x_columns' and
        'init_x_columns_err' respectively. If any color computation is requested
        then new columns will be added to them by 'self.compute_colors()'
        ''' 
        self.init_x_columns = copy.deepcopy(init_x_columns)        
        self.init_x_columns_err = copy.deepcopy(init_x_columns_err)        
        self.x_columns = copy.deepcopy(init_x_columns)
        self.x_columns_err = copy.deepcopy(init_x_columns_err)
        self.x_columns_err_clean = [x for x in self.x_columns_err if x is not None]
        self.mag_columns_4color1 = copy.deepcopy(mag_columns_4color1)
        self.mag_columns_4color2 = copy.deepcopy(mag_columns_4color2)
        self.mag_columns_4color1_err = mag_columns_4color1_err
        self.mag_columns_4color2_err = mag_columns_4color2_err        
        self.y_column = copy.deepcopy(y_column)

        '''
        If 'mag_columns_4color1' and 'mag_columns_4color2' are provided, colors
        will be computed.
        '''
        if self.y_column is None:
            self.output_data = copy.deepcopy(self.input_data[self.x_columns +
                                                             self.x_columns_err_clean])
        else:
            self.output_data = copy.deepcopy(self.input_data[[self.y_column] +
                                                             self.x_columns +
                                                             self.x_columns_err_clean])    
        if (len(mag_columns_4color1) == len(mag_columns_4color2)) and (0<len(mag_columns_4color1)):
            self.compute_colors()

        if valid_values_only: 
            '''
            If exluding rows with missing_value is requested, the data map to 
            a 'output_data_map' and a subsample is selected.
            '''
            self.mapping_output_data()
            self.derive_subsample()
            
        return self.output_data
        
    def mapping_output_data(self):
        self.output_data_map = self.output_data.copy()

        self.output_data_map[self.output_data != self.missing_value] = 1   

        self.output_data_map[self.output_data == self.missing_value] = 0


    def compute_colors(self):
        for i, mag1 in enumerate(self.mag_columns_4color1):
            
            mag2 = self.mag_columns_4color2[i]
            color = mag1 + "-" + mag2            
            self.x_columns.append(color)            
            self.output_data[color] = self.input_data[mag1] - self.input_data[mag2]
            
            '''
            Any color with at least one missed magnitude is set to 
            'self.missing_value' (if not assigned by user, = -99.0 by default)
            '''
            _flag = (self.input_data[mag1] == self.missing_value) |\
                    (self.input_data[mag2] == self.missing_value)
            self.output_data.loc[_flag,color] = self.missing_value

            if self.mag_columns_4color1_err != []:
                color_err = mag1 + "-" + mag2 + "_err"
                self.x_columns_err.append(color_err)
                self.x_columns_err_clean.append(color_err)
                
                self.output_data[color_err] = np.sqrt(self.input_data[self.mag_columns_4color1_err[i]]**2 + 
                                                      self.input_data[self.mag_columns_4color2_err[i]]**2)
                
                '''
                Any color error with at least one missed magnitude error is set to 
                'self.missing_value' (if not assigned by user, = -99.0 by default)
                '''
                _flag = (self.input_data[self.mag_columns_4color1_err[i]] == self.missing_value) |\
                        (self.input_data[self.mag_columns_4color1_err[i]] == self.missing_value)
                self.output_data.loc[_flag,color_err] = self.missing_value
            
            
    def derive_subsample(self):
        '''
        This function selects a subsample for which all targeted columns have 
        acceptable value (axcludes the rows with 'missing_value')
        '''
        if self.y_column is None:
            _num_valid_col = len(self.x_columns) + len(self.x_columns_err_clean)
        else:
            _num_valid_col = len(self.x_columns) + len(self.x_columns_err_clean) + 1 # +1 is for taking into account the existance of y column in data
        self.flag = ( np.sum(self.output_data_map,axis=1) == _num_valid_col )
        self.output_data = self.output_data[self.flag]
         

def data_partition_cc(data, train_frac = 0.7, valid_frac = 0.0, test_frac = 0.3, 
                      grouping_column = None, random_seed = None, return_boolean = False):
    """
    A function to part a given data.Frame into train, validation and test
    samples.
    
    Example:
    Data:
        c1    c2    ID
    0   1.2   'a'   1    
    1   0.4   'z'   1
    2   0.9   'h'   2
    3   9.1   'a'   3
    4   5.8   'j'   3
    ....

    For above data, by grouping_column = 'ID' in calling the function we can force
    the function to put rows 0,1 randomly in the same subsamples.
    
    Parameters
    ----------
    data : pandas.DataFrame
        input data frame in format of pandas.DataFrame
        
    train_frac : float
        It should be between 0 and 1 and represents a fraction of 'data' rows 
        that be allocated for train sample.  By default, the value is set to 0.7. 
        
    valid_frac : float
        It should be between 0 and 1 and represents a fraction of 'data' rows 
        that be allocated for validation sample.  By default, the value is set to 0. 

    test_frac : float
        It should be between 0 and 1 and represents a fraction of 'data' rows 
        that be allocated for test sample.  By default, the value is set to 0.3. 
        
    grouping_column : string, optional
        Name of column that is used to group rows together in the same subsample.
        If the 'grouping_column' is given, the input 'data' must be sorted 
        according to 'grouping_column'.
        
    random_seed : int, optional
        If the random seed is set to value then the ouput will be reproducable.
        
    return_boolean : boolean
        If True, the function returns three list of booleans for selecting each
        row of 'data' for train, validation and test sample. The data frame 
        has three columns and the same number of rows as 'data'. The column names 
        are ['train', 'valid', 'test'].
        If False, the function return three pandas.DataFrame for train, validation
        and test samples. False is the default value.

    Returns
    -------    
    Three pandas.DataFrame objects: respectively train, validation and test
    samples.
    
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if grouping_column is None: 
        _size = len(data)
        _flag = np.random.choice(a = [0, 1, 2], size = _size, p=[train_frac, valid_frac, test_frac])
    else:
        data = data.sort_values(by=[grouping_column])
        _cc = data[grouping_column].values
        unique, counts = np.unique(_cc, return_counts=True)
        _size = len(unique)
        _init_flag = np.random.choice(a = [0, 1, 2], size = _size, p=[train_frac, valid_frac, test_frac])
        _flag = np.repeat(_init_flag, counts)
        
    if return_boolean:
        return _flag == 0, _flag == 1, _flag == 2
    
    else:
        train_data = copy.deepcopy(data[_flag == 0])
        valid_data = copy.deepcopy(data[_flag == 1])
        test_data  = copy.deepcopy(data[_flag == 2])
        return train_data, valid_data, test_data 

           