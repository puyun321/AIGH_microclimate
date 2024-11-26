# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 23:19:50 2021

@author: steve
"""


import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(__file__))
#%% read data
cwb_combine=pd.read_csv("dataset/伸港Gh/2021 dataset/CWB_data/cwb_combine(clean).csv",index_col=0)
cwb_data=cwb_combine.iloc[:,2:]
indoor=pd.read_csv("dataset/伸港Gh/2021 dataset/indoor_data/indoor_hourly(clean).csv",index_col=0)
indoor_data=indoor.iloc[:,1:]
outdoor=pd.read_csv("dataset/伸港Gh/2021 dataset/outdoor_data/outdoor_hourly(clean).csv",index_col=0)
outdoor_data=outdoor.iloc[:,2:]
operation=pd.read_csv("dataset/伸港Gh/2021 dataset/operation/interpolated_data.csv",index_col=0)

def find_feature_num(dataset,feature_num):
    factor=[[]*1 for i in range(0,feature_num)]
    index=1;array=np.asarray(dataset)
    for i in range(0,len(array[0,:])):

        if i>0:
            if index%(feature_num+1)==0:
                index=1
                
            if  index%(feature_num+1)==index:
                factor[index-1].append(i)
                index=index+1
                continue
        else:
            if  index%(feature_num+1)==index:
                factor[index-1].append(i)
                index=index+1
                continue  
    return factor

cwb_factor=np.asarray(find_feature_num(cwb_data,8))

#%% get data statistics
# cwb statistics
cwb_factor_name=cwb_data.columns[0:8]
cwb_statistics=[[]*1 for i in range(0,len(cwb_factor))]
for i in range(0,len(cwb_factor)):
    for j in range(0,len(cwb_factor[i,:])):
        cwb_statistics[i].append(['cwb_%s_mean_t+%s'%(cwb_factor_name[i],j+1),np.mean(cwb_data.iloc[:,cwb_factor[i,j]].to_numpy().flatten())])
        cwb_statistics[i].append(['cwb_%s_std_t+%s'%(cwb_factor_name[i],j+1),np.std(cwb_data.iloc[:,cwb_factor[i,j]].to_numpy().flatten())])
        cwb_statistics[i].append(['cwb_%s_q1_t+%s'%(cwb_factor_name[i],j+1),np.quantile(cwb_data.iloc[:,cwb_factor[i,j]].to_numpy().flatten(),0.25)])
        cwb_statistics[i].append(['cwb_%s_q3_t+%s'%(cwb_factor_name[i],j+1),np.quantile(cwb_data.iloc[:,cwb_factor[i,j]].to_numpy().flatten(),0.75)])    
    cwb_statistics[i]=np.asarray(cwb_statistics[i])
    
cwb_statistics=np.concatenate(cwb_statistics,axis=1)   
    
# outdoor statistics
outdoor_factor_name=outdoor_data.columns
outdoor_statistics=[[]*1 for i in range(0,len(outdoor_factor_name))]
for i in range(0,len(outdoor_factor_name)):
    outdoor_statistics[i].append(['outdoor_%s_mean'%outdoor_factor_name[i],np.mean(outdoor_data.iloc[:,i])])
    outdoor_statistics[i].append(['outdoor_%s_std'%outdoor_factor_name[i],np.std(outdoor_data.iloc[:,i])])
    outdoor_statistics[i].append(['outdoor_%s_q1'%outdoor_factor_name[i],np.quantile(outdoor_data.iloc[:,i],0.25)])
    outdoor_statistics[i].append(['outdoor_%s_q3'%outdoor_factor_name[i],np.quantile(outdoor_data.iloc[:,i],0.75)])    
outdoor_statistics=np.concatenate(outdoor_statistics,axis=0)

# indoor statistics
indoor_factor_name=indoor_data.columns
indoor_statistics=[[]*1 for i in range(0,len(indoor_factor_name))]
for i in range(0,len(indoor_factor_name)):
    indoor_statistics[i].append(['indoor_%s_mean'%indoor_factor_name[i],np.mean(indoor_data.iloc[:,i])])
    indoor_statistics[i].append(['indoor_%s_std'%indoor_factor_name[i],np.std(indoor_data.iloc[:,i])])
    indoor_statistics[i].append(['indoor_%s_q1'%indoor_factor_name[i],np.quantile(indoor_data.iloc[:,i],0.25)])
    indoor_statistics[i].append(['indoor_%s_q3'%indoor_factor_name[i],np.quantile(indoor_data.iloc[:,i],0.75)])    
indoor_statistics=np.concatenate(indoor_statistics,axis=0)





    
    
    