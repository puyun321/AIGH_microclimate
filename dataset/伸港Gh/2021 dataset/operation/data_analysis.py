# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:14:13 2021

@author: steve
"""

import pandas as pd
import numpy as np
import copy

operation_data=pd.read_csv("D:/database/溫室/伸港Gh/operation/interpolated_data.csv",index_col=0)
indoor=pd.read_csv("D:/database/溫室/伸港Gh/indoor_data/indoor_10min(clean).csv",index_col=0)
outdoor=pd.read_csv("D:/database/溫室/伸港Gh/outdoor_data/outdoor_10min(clean).csv",index_col=0)

index_5=pd.read_excel("D:/database/溫室/伸港Gh/operation/interpolate_index.xlsx",sheet_name='index_5',index_col=0)
index_4=pd.read_excel("D:/database/溫室/伸港Gh/operation/interpolate_index.xlsx",sheet_name='index_4',index_col=0)
index_3=pd.read_excel("D:/database/溫室/伸港Gh/operation/interpolate_index.xlsx",sheet_name='index_3',index_col=0)
index_2=pd.read_excel("D:/database/溫室/伸港Gh/operation/interpolate_index.xlsx",sheet_name='index_2',index_col=0)

#%%
operation_data_5=operation_data.iloc[index_5.iloc[:,0],:]
timestep_5=operation_data_5.iloc[:,0]
timestep_5 = [timestep_5.iloc[i][-5:] for i in range(0,len(timestep_5))]
only_time_5=[]
for i in range(0,len(timestep_5)):
    store=''.join(character  for character in timestep_5[i] if character.isalnum())
    only_time_5.append(store)
only_time_5=pd.Series(only_time_5)
unique_time=only_time_5.drop_duplicates(keep='first')

def find_time_index(time_series,unique_time_series):
    array=copy.deepcopy(time_series);unique_array=copy.deepcopy(unique_time_series)
    unique_array_index=[[]*1 for i in range(0,len(unique_array))]
    for i in range(0,len(unique_array)):
        unique_array_index[i]=array[array==unique_array[i]].index
    return unique_array_index

unique_time_index=find_time_index(only_time_5,unique_time)

#%%
unique_operation_data_5=operation_data.iloc[index_5.iloc[:,0],1:-1].drop_duplicates(keep='first')
unique_outdoor_5=outdoor.iloc[unique_operation_data_5.index,:]

def find_array_index(unique_operation,operation):
    unique_array=copy.deepcopy(unique_operation);target_array=copy.deepcopy(operation)
    unique_array=np.asarray(unique_array);target_array=np.asarray(target_array)
    unique_index=[[]*1 for i in range(0,len(unique_array))]
    for i in range(0,len(unique_array)):
        for j in range(0,len(target_array)):
            if (target_array[j,:]==unique_array[i,:]).all()==True:
                unique_index[i].append(j)
    return unique_index
def mean_array(time_index,target_data):
    index_array=copy.deepcopy(time_index);target_array=copy.deepcopy(target_data)
    index_array=np.asarray(index_array);target_array=np.asarray(target_array)
    mean_output=[[]*1 for i in range(0,len(index_array))]
    for i in range(0,len(index_array)):
        mean_output[i]=np.mean(target_array[np.asarray(index_array[i]),:],axis=0)
    mean_output=np.asarray(mean_output)
    return mean_output

unique_operation_index=find_array_index(unique_operation_data_5,operation_data_5.iloc[:,1:-1])
unique_outdoor_data=mean_array(unique_operation_index,outdoor.iloc[:,1:-1])
unique_indoor_data=mean_array(unique_operation_index,indoor.iloc[:,1:-1])

#%%
"""
read analysis excel
"""
from sklearn.ensemble import RandomForestRegressor
# Temperature
analysis_data=pd.read_excel("D:/database/溫室/伸港Gh/operation/operation_data_analysis.xlsx")
#%%
#change condition in this row
# index=analysis_data[analysis_data.loc[:,'Temp']<analysis_data.iloc[0,14]].index 
# index=analysis_data[(analysis_data.loc[:,'Temp']<analysis_data.iloc[1,14]) & (analysis_data.loc[:,'Temp']>analysis_data.iloc[0,14])].index 
# index=analysis_data[(analysis_data.loc[:,'Temp']<analysis_data.iloc[2,14]) & (analysis_data.loc[:,'Temp']>analysis_data.iloc[1,14])].index 
index=analysis_data[analysis_data.loc[:,'Temp']>analysis_data.iloc[2,14]].index

#%%
forest = RandomForestRegressor()
forest.fit(analysis_data.iloc[index,0:6], analysis_data.iloc[index,7])
importances = forest.feature_importances_

forest.fit(analysis_data.iloc[index,0:6], analysis_data.iloc[index,-4])
importances = forest.feature_importances_

#%%
#RH
# index=analysis_data[analysis_data.loc[:,'RH']<analysis_data.iloc[0,15]].index #change condition in this row
# index=analysis_data[(analysis_data.loc[:,'RH']<analysis_data.iloc[1,15]) & (analysis_data.loc[:,'RH']>analysis_data.iloc[0,15])].index 
# index=analysis_data[(analysis_data.loc[:,'RH']<analysis_data.iloc[2,15]) & (analysis_data.loc[:,'RH']>analysis_data.iloc[1,15])].index 
index=analysis_data[analysis_data.loc[:,'RH']>analysis_data.iloc[2,15]].index

#%%
forest = RandomForestRegressor()
forest.fit(analysis_data.iloc[:,0:6], analysis_data.iloc[index,8])

# forest.fit(analysis_data.iloc[index,0:6], analysis_data.iloc[index,8])
importances = forest.feature_importances_

#%%
#PAR
# index=analysis_data[analysis_data.loc[:,'PAR']<analysis_data.iloc[0,16]].index #change condition in this row
# index=analysis_data[(analysis_data.loc[:,'PAR']<analysis_data.iloc[1,16]) & (analysis_data.loc[:,'PAR']>analysis_data.iloc[0,16])].index 
# index=analysis_data[(analysis_data.loc[:,'PAR']<analysis_data.iloc[2,16]) & (analysis_data.loc[:,'PAR']>analysis_data.iloc[1,16])].index 
index=analysis_data[analysis_data.loc[:,'PAR']>analysis_data.iloc[1,16]].index

#%%
forest = RandomForestRegressor()
# forest.fit(analysis_data.iloc[:,0:6], analysis_data.iloc[:,9])

forest.fit(analysis_data.iloc[index,0:6], analysis_data.iloc[index,8])
importances = forest.feature_importances_