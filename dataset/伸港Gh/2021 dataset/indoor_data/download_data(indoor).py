# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 22:41:22 2021

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
comdText="SELECT * FROM dbo.shgindoor WHERE TimeStep>='2020040100' "
cursor.execute(comdText)
rows = cursor.fetchall()
dataset = np.asarray(rows)
dataset = pd.DataFrame(dataset)
dataset.columns=['Time','RNO','Voltage','PTemp','AirTemp','RH','PAR','SHF','NR','VWF','VWM','VWB','Eb1','EC1','SoilTemp1','VWC1','Eb2','EC2','SoilTemp2','VWC2','TimeStep']
selected_columns=[0,4,5,6,7,8,9,10,11,20]
selected_dataset = dataset.iloc[:,selected_columns]
#%%
selected_dataset.iloc[:,1:] = (selected_dataset.iloc[:,1:]).astype(float)
#%%
#10min
grouped_min=selected_dataset.groupby(pd.Grouper(key='Time',freq='10T')).mean()
datetime_min=grouped_min.index
grouped_min.to_csv("D:/database/溫室/伸港Gh/indoor_data/indoor_10min(fulldate).csv")

#%%
#hourly
grouped=selected_dataset.groupby(pd.Grouper(key='Time',freq='h')).mean()
grouped_new=grouped.reset_index(drop=True)
error_index=(grouped_new[grouped_new.iloc[:,1].isnull()]).index.tolist()
grouped.to_csv("D:/database/溫室/伸港Gh/indoor_data/indoor_hourly(fulldate).csv")
pd.DataFrame(error_index).to_csv("D:/database/溫室/伸港Gh/indoor_data/error_index.csv")

