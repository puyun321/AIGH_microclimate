# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:16:41 2021

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
comdText="SELECT * FROM dbo.WeatherReport WHERE TimeStep>='2020040100' "
cursor.execute(comdText)
rows = cursor.fetchall()
dataset = np.asarray(rows)
dataset = pd.DataFrame(dataset)

#%%
col_name=['StationID','Time','Stathpa','Seahpa','Temp','DewTemp','RH','WaterhPa','AvrWindSpeed','AvrWindDir','MaxWindSpeed','MaxWindDir','IntWindSpeed','IntWindDir','Rainfall','SunHr','PAR','TimeStep']
dataset.columns = col_name
sorted_data=dataset.sort_values(by=['TimeStep'])
sorted_data=sorted_data.reset_index(drop=True)
sorted_data=sorted_data.iloc[:,1:]
