# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:52:54 2021

@author: steve
"""


import numpy as np
import pandas as pd
from functools import reduce
import pyodbc

#%%
#download data from database
#Open Database
conn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=140.112.183.63;DATABASE=IOTDB;UID=sa;PWD=Fj1957')
cursor = conn.cursor()
comdText="SELECT * FROM dbo.Shield WHERE TimeStep>='2020040100' "
cursor.execute(comdText)
rows = cursor.fetchall()
dataset = np.asarray(rows)
dataset = pd.DataFrame(dataset)
dataset.columns=['time','Skykight','InShade','NorthUp','Northdown','SouthUp','Southdown','GHID','TimeStep']
sorted_data=dataset.sort_values(by=['TimeStep'])
sorted_data=sorted_data.reset_index(drop=True)
selected_columns=[0,1,2,3,4,5,6,8]
sorted_data= sorted_data.iloc[:,selected_columns]
sorted_data.iloc[:,1:]=sorted_data.iloc[:,1:].astype('float')

#%%
grouped_min=sorted_data.groupby(pd.Grouper(key='time',freq='10T')).mean()
datetime_min=grouped_min.index
grouped_min.to_csv("D:/database/溫室/伸港Gh/operation/operation_10min(fulldate).csv")
