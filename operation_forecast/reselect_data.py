# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:32:26 2021

@author: steve
"""

import pandas as pd
import numpy as np
import math
import datetime
from functools import reduce
"""
read data from local databse
"""
#%%
#data preprocessing function
def only_datetime(datetime_):
    # datetime_=datetime.datetime.strptime(datetime_,'%Y%m%d%H%M%S').astype(str)

    int_datetime=[[]*1 for i in range(0,len(datetime_))]
    check=str(datetime_[0])
    for i in range(0,len(datetime_)):
        print(i)
        if len(check)>16:
            try:
                current_datetime=str(datetime.datetime.strptime(datetime_[i],'%Y/%m/%d %H:%M:%S')) 
            except:
                current_datetime=str(datetime.datetime.strptime(datetime_[i],'%Y-%m-%d %H:%M:%S'))                 
        else:
            current_datetime=str(datetime.datetime.strptime(datetime_[i],'%Y/%m/%d %H:%M'))
        datetime_place=[0,1,2,3,5,6,8,9,11,12,14,15]
        # if len(current_datetime)==16 or len(current_datetime)>16:
        #     datetime_place=[0,1,2,3,5,6,8,9,11,12,14,15]
        # elif len(current_datetime)==15:
        #     if current_datetime[6]=='/':
        #         datetime_place=[0,1,2,3,5,7,8,10,11,13,14]
        #     else:
        #         datetime_place=[0,1,2,3,5,6,8,10,11,13,14]                
        # elif len(current_datetime)==14 or len(current_datetime)<14:
        #     datetime_place=[0,1,2,3,5,7,9,10,12,13]        
        for j in range(0,len(datetime_place)):
            int_datetime[i].append(current_datetime[datetime_place[j]])
        int_datetime[i]="".join(int_datetime[i])
    int_datetime=np.asarray(int_datetime)
    return int_datetime

#%%
#read 10min operation data
operation_10min=pd.read_csv("D:/database/溫室/伸港Gh/operation/interpolated_data.csv",index_col=0)
#read 10min cwb data
operation_10min.loc[:,'time']=pd.to_datetime(operation_10min.loc[:,'time'])
operation_datetime=operation_10min.iloc[:,0].astype(str)
operation_datetime=only_datetime(operation_datetime).astype(np.int64)

#%%
#read cwb data
# cwb_t0=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t0.csv")
cwb_t1=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t1.csv")
cwb_t2=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t2.csv")
cwb_t3=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t3.csv")
cwb_t4=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t4.csv")
cwb_t5=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t5.csv")
cwb_t6=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t6.csv")
cwb_datetime=np.asarray([str(cwb_t1.loc[i,'Report_time'])+'00' for i in range(0,len(cwb_t1))]).astype(np.int64)
selected_columns=[5,6,7,8,9,10,11,12]
cwb_combine=(pd.concat([cwb_t1.iloc[:,1:3],cwb_t1.iloc[:,selected_columns],cwb_t2.iloc[:,selected_columns],cwb_t3.iloc[:,selected_columns],cwb_t4.iloc[:,selected_columns],cwb_t5.iloc[:,selected_columns],cwb_t6.iloc[:,selected_columns]],axis=1)).reset_index(drop=True)

#%%
#negative value change to 0
def remove_negative(dataframe):
    dataframe[dataframe<0]=0
    return dataframe
    
cwb_combine.iloc[:,2:]=remove_negative(cwb_combine.iloc[:,2:])

#%%
#no error index#
#read error indices 
# outdoor_error_index=pd.read_csv("D:/database/溫室/伸港Gh//outdoor_data/error_index.csv",index_col=0)
# indoor_error_index=pd.read_csv("D:/database/溫室/伸港Gh//indoor_data/error_index.csv",index_col=0)
# error_index=np.union1d(outdoor_error_index,indoor_error_index)

#%%
#read iot outdoor data
outdoor_complete_10min=pd.read_csv("D:/database/溫室/伸港Gh/outdoor_data/outdoor_10min(clean).csv",index_col=0)
outdoor_complete_10min.iloc[:,1:]=remove_negative(outdoor_complete_10min.iloc[:,1:])
outdoor_datetime=only_datetime(np.asarray(outdoor_complete_10min.loc[:,'Time'])).astype(np.int64)
max_cwb_datetime=int(str(max(cwb_datetime)))
outdoor_datetime=np.asarray([element for index,element in enumerate(outdoor_datetime) if element<=max_cwb_datetime])

#%%
#read iot indoor data
indoor_complete_10min=pd.read_csv("D:/database/溫室/伸港Gh/indoor_data/indoor_10min(clean).csv",index_col=0)
indoor_complete_10min.iloc[:,1:]=remove_negative(indoor_complete_10min.iloc[:,1:])
indoor_datetime=only_datetime(np.asarray(indoor_complete_10min.loc[:,'Time'])).astype(np.int64)
max_cwb_datetime=int(str(max(cwb_datetime)))
indoor_datetime=np.asarray([element for index,element in enumerate(indoor_datetime) if element<=max_cwb_datetime])

#%%
#same index operation, cwb, outdoor, indoor 
same_element=reduce(np.intersect1d, (cwb_datetime,outdoor_datetime,indoor_datetime))
# same_element=reduce(np.intersect1d, (operation_datetime,cwb_datetime,outdoor_datetime,indoor_datetime))

# same_index_operation=np.squeeze(np.asarray([np.argwhere(operation_datetime==same_element[i]) for i in range(0,len(same_element))]))
same_index_cwb=np.squeeze(np.asarray([np.argwhere(cwb_datetime==same_element[i]) for i in range(0,len(same_element))]))
same_index_outdoor=np.squeeze(np.asarray([np.argwhere(outdoor_datetime==same_element[i]) for i in range(0,len(same_element))]))
same_index_indoor=np.squeeze(np.asarray([np.argwhere(indoor_datetime==same_element[i]) for i in range(0,len(same_element))]))

#%%
#create forecast dataset (t+1 to t+18)
outdoor_output=[];indoor_output=[]
timestep=18
for i in range(0,timestep+1):
    if i!=timestep:
        outdoor_output.append(outdoor_complete_10min.iloc[i:-(timestep-i),1:-1].reset_index(drop=True))
        indoor_output.append(indoor_complete_10min.iloc[i:-(timestep-i),1:-1].reset_index(drop=True))        
    else:
        outdoor_output.append(outdoor_complete_10min.iloc[i:,1:-1].reset_index(drop=True))   
        indoor_output.append(indoor_complete_10min.iloc[i:,1:-1].reset_index(drop=True))   

index_2_remove=pd.DataFrame([i for i in range(len(outdoor_complete_10min)-timestep,len(outdoor_complete_10min))])
outdoor_output=pd.concat(outdoor_output,axis=1)
indoor_output=pd.concat(indoor_output,axis=1)

#%%
writer = pd.ExcelWriter('D:/research/溫室/伸港/operation_forecast/selected_data/same_index(10min).xlsx', engine='xlsxwriter')
#save same index for cwb,outdoor,indoor  
# pd.DataFrame(same_index_operation).to_excel(writer,sheet_name="operation_index")
pd.DataFrame(same_index_cwb).to_excel(writer,sheet_name="cwb_index")
pd.DataFrame(same_index_outdoor).to_excel(writer,sheet_name="outdoor_index")
pd.DataFrame(same_index_indoor).to_excel(writer,sheet_name="indoor_index")
writer.save()
cwb_combine.to_csv("D:/research/溫室/伸港/operation_forecast/selected_data/cwb_combine(clean)-10min.csv")

indoor_output.to_csv("D:/research/溫室/伸港/operation_forecast/selected_data/indoor_10min_output(clean).csv")
outdoor_output.to_csv("D:/research/溫室/伸港/operation_forecast/selected_data/outdoor_10min_output(clean).csv")
index_2_remove.to_csv("D:/research/溫室/伸港/operation_forecast/selected_data/extra_remove_index(10min).csv")


