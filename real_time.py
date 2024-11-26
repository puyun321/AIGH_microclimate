# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:27:28 2024

@author: Steve
"""

import os
import numpy as np
import pandas as pd
import datetime
os.chdir(os.path.dirname(__file__))

#%% data preprocessing function
#normalization
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

#reshape input data as 2d
def convo_input(dataset,timestep,same_index,error_index=0):
    array=np.asarray(dataset);index=np.asarray(error_index);same_index=np.squeeze(np.asarray(same_index))
    convo_dataset=[[]*1 for i in range(0,len(array)+1)]
    if type(error_index)==int: #if no error_indices
        for i in range(timestep,len(array)+1):
            convo_dataset[i].append(array[i-timestep:i,:])
        convo_dataset=list(filter(None,convo_dataset))
    else:
        for i in range(timestep,len(array)+1):
            if i in index:
                continue
            else:
                convo_dataset[i].append(array[i-timestep:i,:])
        convo_dataset=list(filter(None,convo_dataset))
    convo_dataset=np.squeeze(np.asarray([convo_dataset[same_index[j]-(timestep-1)] for j in range(int(timestep/6),len(same_index))]))
    return convo_dataset

# arrange cwb dataset into 2d input data
def cwb_data_arrangement(dataset,feature_num):
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
    
    arranged_cwb_data=[[]*1 for i in range(0,len(array))]
    for i in range(0,len(array)):
        for j in range(0,len(factor)):
            selected_array=np.squeeze(np.asarray([array[i,f] for f in factor[j]]))
            arranged_cwb_data[i].append(selected_array)
        arranged_cwb_data[i]=np.transpose(np.asarray(arranged_cwb_data[i]))
    arranged_cwb_data=np.squeeze(np.asarray(arranged_cwb_data))

    return arranged_cwb_data,factor

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

def operation_convo(dataset,timestep,same_index):
    array=np.asarray(dataset);same_index=np.squeeze(np.asarray(same_index))
    convo_dataset=[[]*1 for i in range(0,len(array)+1)]

    for i in range(int(timestep/6),len(same_index)):
        convo_dataset[i].append(array[same_index[i]-timestep:same_index[i],:])
    convo_dataset=list(filter(None,convo_dataset))
    convo_dataset=np.squeeze(np.asarray(convo_dataset))
    # convo_dataset=np.squeeze(np.asarray(convo_dataset)[same_index[1:]-(timestep-1)])
    return convo_dataset   

#%% physically based model function
def physically_based_model(operation, Temp_GH0, Temp_GH6, Temp_GHout, PAR_GH, datetime_):
    
    Cover_area = [1560,187,125,187,125]
    GH_vol = 9360; step = 18    #unit: m^3, t+1-t+18
    
    n_par = 0.368; n_photon = 4.56; t_crit = 43.0
    s_crit = 1.0; k_ir_1 = 0.3; k_ir_5 = 1.0; k_ir = k_ir_1
    
    h_air = 100    #unit: W/m^2.K
    k_cover = 0.5    #unit: W/m.K
    l_cover = 0.00015    #unit: m
    k_shed = 0.5; l_shed = 0.0002
    
    # Variables Setting 
    Temp_GH_esti = []; 
    Shed_ratio = operation/100    

    factor_1=float(datetime_[11:13])
    factor_2=float(datetime_[14:16])/60
    TF = factor_1+factor_2 # time factor

    Temp_slope = (Temp_GH0-Temp_GH6) / step
    if((Temp_GH0 >= t_crit) | (Temp_slope >= s_crit)):
        k_ir = k_ir_5
    elif((TF >= 7) & (TF <= 16)):
        k_ir = 0.0081*TF*TF - 0.2154*TF + 1.6984
    else:
        k_ir = k_ir_1
            
    cover_column=[0,2,3,4,5]
    Cover_Ir = 0; Cover_U = 0
    
    for i in range(len(Cover_area)):
        Cover_Ir += Cover_area[i] * (1-Shed_ratio[cover_column[i]])
        if(Shed_ratio[1] <= 0):
            U = 1 / (1/h_air + l_cover/k_cover + 1/h_air)
        else:
            U1 = 1 / (1/h_air + l_cover/k_cover + 1/h_air)
            U2 = 1 / (1/h_air + l_cover/k_cover + l_shed/k_shed + 1/h_air)
            U = Shed_ratio[cover_column[i]]*U2 + (1-Shed_ratio[cover_column[i]])*U1
        Cover_U += U * Cover_area[i]
    
    # Temperature Modeling
    for i in range(step):
        Ir = PAR_GH[i] / (n_par*n_photon)
        Q_sun = Ir * Cover_Ir * k_ir
        Temp_GH = Temp_GHout[i] + Q_sun/Cover_U
        Temp_GH_esti.append(Temp_GH)
    
    return Temp_GH_esti


#%% read data

# change working directory
timestep=18
database_path='dataset/伸港Gh/2021 dataset'
os.chdir(database_path)

# read data
cwb_combine=pd.read_csv("CWB_data/cwb_combine(clean).csv",index_col=0)
indoor=pd.read_csv("indoor_data/indoor_10min(clean).csv",index_col=0)
outdoor=pd.read_csv("outdoor_data/outdoor_10min(clean).csv",index_col=0)
operation=pd.read_csv("operation/interpolated_data.csv",index_col=0)

# get min and max value for normalization
max_temp = max(indoor.loc[:,'AirTemp']); min_temp = min(indoor.loc[:,'AirTemp'])
max_rh = max(indoor.loc[:,'RH']); min_rh = min(indoor.loc[:,'RH'])
max_par = max(indoor.loc[:,'PAR']); min_par = min(indoor.loc[:,'PAR'])

#read same index
same_cwb_index=pd.read_excel('same_index(10min).xlsx',sheet_name="cwb_index",index_col=0)
same_outdoor_index=pd.read_excel('same_index(10min).xlsx',sheet_name="outdoor_index",index_col=0)
same_indoor_index=pd.read_excel('same_index(10min).xlsx',sheet_name="indoor_index",index_col=0)
same_operation_index=pd.read_excel('same_index(10min).xlsx',sheet_name="operation_index",index_col=0)

#normalization
norm_cwb = normalize(cwb_combine.iloc[:,2:])
norm_indoor = normalize(indoor.iloc[:,1:-1])
norm_outdoor = normalize(outdoor.iloc[:,1:-1])

#make sure the all input time horizon is aligned
operation_input=operation_convo(operation.iloc[:,1:],timestep,same_operation_index)
cwb_convo_input=norm_cwb.iloc[np.squeeze(np.asarray(same_cwb_index)),:]
selected_cwb_convo_input,factor=cwb_data_arrangement(cwb_convo_input.iloc[int(timestep/6):,:],8)
indoor_convo_input=convo_input(norm_indoor,timestep,same_indoor_index)
outdoor_convo_input=convo_input(norm_outdoor,timestep, same_indoor_index)

#select indoor data according factor: AirTemp,RH,PAR,SHF,NR,VMF,VWM,VWB
selected_indoor_factor=np.asarray([0,1,2,3,4,5,6,7])
selected_indoor_convo_input=indoor_convo_input[:,:,np.asarray(selected_indoor_factor)]

#select outdoor data according factor: OutTemp,OutRH,OutPAR,WindSpeed,WindDir
selected_outdoor_factor=np.asarray([0,1,2])
selected_outdoor_convo_input=outdoor_convo_input[:,:,np.asarray(selected_outdoor_factor)]

#select  operation data factor: Skykight, Inshade, Northup, Northdown, Southup, Southdown
selected_operation_factor=np.asarray([0,1,2,3,4,5])
selected_operation_convo_input=operation_input[:,:,np.asarray(selected_operation_factor)]/100

# t-18 to t observation datetime, temperature and par 
indoor_obs=convo_input(indoor,timestep,same_indoor_index)
datetime_ = indoor_obs[:,0,0]
Temp_GH = indoor_obs[:,:,1] #t to t-18 grenhouse indoor par

#%% load deep learning model and forecast
os.chdir(os.path.join(os.getcwd(), r'..\..\..'))
from tensorflow.keras.models import load_model

batch_size=32

# load greenhouse indoor model and predict
GHin_path=r"direct_forecast\indoor(CWB)\model_program\model\cnn-lstm(stateless).hdf5"
GHin_model=load_model(GHin_path)
GHin_predict=GHin_model.predict([selected_cwb_convo_input, selected_operation_convo_input], batch_size=batch_size)
input_factor=np.asarray([0,1,2]) 
feature_num=len(input_factor)
GHin_feature_index=find_feature_num(GHin_predict,feature_num) 

# load greenhouse outdoor model and predict
GHout_path=r"direct_forecast\outdoor(CWB)\model_program\model\cnn-lstm(stateless).hdf5"
GHout_model=load_model(GHout_path)
GHout_predict=GHout_model.predict(selected_cwb_convo_input, batch_size=batch_size)
output_factor=np.asarray([0,1,2])
feature_num=len(output_factor)
GHout_feature_index=find_feature_num(GHout_predict,feature_num) 

# physically-based model prediction
if len(GHout_predict.shape)>1:
    operation = selected_operation_convo_input[:,0,:]
    Temp_GH0 = Temp_GH[:,0] #timestep t temperature
    Temp_GHout = GHout_predict[:,GHout_feature_index[0][1:]] # timestep t+1 to t+18 temperature
    Temp_GH6 = Temp_GH[:,5] #timestep t-5 temperature
    PAR_GH = GHin_predict[:,GHin_feature_index[2][1:]] # timestep t+1 to t+18 temperature
    
    GHin_simulation=[physically_based_model(operation[i,:], Temp_GH0[i], Temp_GH6[i], Temp_GHout[i,:], PAR_GH[i,:], datetime_[i],) for i in range(GHout_predict.shape[0])]
    GHin_simulation = np.array(GHin_simulation)

# hybrid model forecast
CWB_GHin_path=r"hybrid_forecast\model\cnn-lstm(cwb).hdf5"
CWB_GHin_model=load_model(CWB_GHin_path)
CWB_GHin_predict=CWB_GHin_model.predict([selected_cwb_convo_input, GHin_simulation], batch_size=batch_size)
CWB_GHin_feature_index=find_feature_num(CWB_GHin_predict,feature_num) #indoor

CWB_GHin_Temp_predict=GHin_predict[:,CWB_GHin_feature_index[0][1:]]
CWB_GHin_Temp_predict=CWB_GHin_Temp_predict*(max_temp-min_temp)+min_temp

CWB_GHin_RH_predict=GHin_predict[:,CWB_GHin_feature_index[1][1:]]
CWB_GHin_RH_predict=CWB_GHin_RH_predict*(max_rh-min_rh)+min_rh

CWB_GHin_PAR_predict=GHin_predict[:,CWB_GHin_feature_index[2][1:]]
CWB_GHin_PAR_predict=CWB_GHin_PAR_predict*(max_par-min_par)+min_par

#%% save result
np.save(r"result\temperature.npy",CWB_GHin_Temp_predict)
np.save(r"result\RH.npy",CWB_GHin_RH_predict)
np.save(r"result\par.npy",CWB_GHin_PAR_predict)
