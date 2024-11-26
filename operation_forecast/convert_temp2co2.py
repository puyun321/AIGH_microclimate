# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 19:35:58 2021

@author: steve
"""
import pandas as pd
import numpy as np

#%%
#read input
indoor=pd.read_csv("D:/database/溫室/伸港Gh/indoor_data/indoor_10min(clean).csv",index_col=0)

#read output
indoor_output=pd.read_csv("D:/research/溫室/伸港/operation_forecast/selected_data/indoor_10min_output(clean).csv",index_col=0)
outdoor_output=pd.read_csv("D:/research/溫室/伸港/operation_forecast/selected_data/outdoor_10min_output(clean).csv",index_col=0)

#read same index
same_cwb_index=pd.read_excel('D:/research/溫室/伸港/operation_forecast/selected_data/same_index(10min).xlsx',sheet_name="cwb_index",index_col=0)
same_outdoor_index=pd.read_excel('D:/research/溫室/伸港/operation_forecast/selected_data/same_index(10min).xlsx',sheet_name="outdoor_index",index_col=0)
same_indoor_index=pd.read_excel('D:/research/溫室/伸港/operation_forecast/selected_data/same_index(10min).xlsx',sheet_name="indoor_index",index_col=0)

#read error index
extra_remove_index=pd.read_csv("D:/research/溫室/伸港/operation_forecast/selected_data/extra_remove_index(10min).csv",index_col=0) 
error_index=np.asarray(extra_remove_index)

#%%
timestep=18
def model_output(dataset,input_timestep,same_index,feature_num,error_index=0):
    array=np.asarray(dataset);index=np.asarray(error_index);same_index=np.squeeze(np.asarray(same_index))
    convo_dataset=[[]*1 for i in range(0,len(dataset))]
    for i in range((input_timestep-1),len(array)):
        if i in index:
            continue
        else:
            convo_dataset[i-(input_timestep-1)].append(array[i,:])
    convo_dataset=list(filter(None,convo_dataset))
    convo_dataset=[convo_dataset[same_index[j]-(input_timestep-1)] for j in range(int(input_timestep/6),len(same_index))]
    convo_dataset=np.squeeze((np.asarray(convo_dataset)))
    
    return convo_dataset
    
indoor_model_output=model_output(indoor_output,timestep,same_indoor_index,8,error_index) #forecast 8 features
outdoor_model_output=model_output(outdoor_output,timestep,same_indoor_index,5,error_index) #forecast 5 features

#%%

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

#select indoor output data according factor AirTemp,RH,PAR,SHF,NR,VMF,VWM,VWB
output_factor=find_feature_num(indoor_output,8) 
indoor_factor=np.asarray(output_factor)
selected_output_factor=[0]
selected_temp_output=np.concatenate([indoor_model_output[:,np.asarray(indoor_factor[selected_output_factor,i])] for i in range(0,timestep+1)],axis=1)
selected_output_factor=[1]
selected_rh_output=np.concatenate([indoor_model_output[:,np.asarray(indoor_factor[selected_output_factor,i])] for i in range(0,timestep+1)],axis=1)
selected_output_factor=[2]
selected_par_output=np.concatenate([indoor_model_output[:,np.asarray(indoor_factor[selected_output_factor,i])] for i in range(0,timestep+1)],axis=1)

output_factor=find_feature_num(outdoor_output,5) 
outdoor_factor=np.asarray(output_factor)
selected_output_factor=[0]
selected_temp_outdoor_output=np.concatenate([outdoor_model_output[:,np.asarray(outdoor_factor[selected_output_factor,i])] for i in range(0,timestep+1)],axis=1)

#%%
#extract date info
indoor_info=pd.DataFrame([indoor.iloc[index,:] for index in range(0,len(indoor)) if index not in error_index])
indoor_info=indoor_info.iloc[np.squeeze(np.asarray(same_indoor_index.iloc[int(timestep/6):,0])),0].reset_index(drop=True)

#%%
datetime=indoor_info
timestep=np.asarray([6,12,18])
Temp_t123=selected_temp_output[:,timestep]
RH_t123=selected_rh_output[:,timestep]
PAR_t123=selected_par_output[:,timestep]
Temp_outdoor_t123=selected_temp_outdoor_output[:,timestep]

#%%
def convert2CO2(temp):
    co2_array=[]
    for i,element in enumerate(temp):
        co2=-4.5274*element+541.58
        co2_array.append(co2)
    co2_array=np.asarray(co2_array)
    return co2_array

co2_t123=pd.DataFrame.transpose(pd.DataFrame([convert2CO2(Temp_t123[:,i]) for i in range(0,len(Temp_t123[0,:]))]))
co2_t123.columns=['T+1','T+2','T+3']

writer = pd.ExcelWriter('D:/research/溫室/伸港/operation_forecast/selected_data/weather_data.xlsx', engine='xlsxwriter')
pd.concat([datetime,co2_t123],axis=1).to_excel(writer,sheet_name="co2_simulate(indoor)")
pd.concat([datetime,pd.DataFrame(Temp_t123)],axis=1).to_excel(writer,sheet_name="Temp_real(indoor)")
pd.concat([datetime,pd.DataFrame(RH_t123)],axis=1).to_excel(writer,sheet_name="RH_real(indoor)")
pd.concat([datetime,pd.DataFrame(PAR_t123)],axis=1).to_excel(writer,sheet_name="PAR_real(indoor)")
pd.concat([datetime,pd.DataFrame(Temp_outdoor_t123)],axis=1).to_excel(writer,sheet_name="Temp_real(outdoor)")
writer.save()
