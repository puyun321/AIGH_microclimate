# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:31:14 2021

@author: steve
"""


import pandas as pd
import numpy as np

outdoor=pd.read_csv("D:/database/溫室/伸港Gh/outdoor_data/outdoor_10min(fulldate).csv")

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_ghdata(array):
    df_array=np.asarray(array).astype(np.float)
    array_mean=np.mean(df_array,axis=0)
    array_std=np.std(df_array,axis=0)
    for i in range(0,len(df_array[0,:])):
        for j in range(0,len(df_array[:,i])):
            if df_array[j,i]>array_mean[i]+3*array_std[i] or df_array[j,i]<array_mean[i]-3*array_std[i]:
                df_array[j,i]=np.nan
        c_array=df_array[:,i]
        nans, x=nan_helper(c_array)
        c_array[nans]= np.interp(x(nans), x(~nans), c_array[~nans].astype(np.float))    
        df_array[:,i]=c_array
    return df_array

clean_data=pd.DataFrame(interpolate_ghdata(outdoor.iloc[:,1:]))

outdoor.iloc[:,1:]=np.asarray(clean_data)

outdoor.to_csv("D:/database/溫室/伸港Gh/outdoor_data/outdoor_10min(clean).csv")