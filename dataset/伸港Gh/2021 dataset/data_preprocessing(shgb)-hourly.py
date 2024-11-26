# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:32:26 2021

@author: steve
"""

import pandas as pd
import numpy as np
import math
from functools import reduce
"""
read data from local databse
"""
#%%
#data preprocessing function
def only_datetime(datetime):
    datetime=datetime.astype(str)
    datetime_place=[0,1,2,3,5,6,8,9,11,12]
    int_datetime=[[]*1 for i in range(0,len(datetime))]
    for i in range(0,len(datetime)):
        current_datetime=datetime[i]
        for j in range(0,len(datetime_place)):
            int_datetime[i].append(current_datetime[datetime_place[j]])
        int_datetime[i]="".join(int_datetime[i])
    int_datetime=np.asarray(int_datetime)
    return int_datetime

# def find_samedatetime(cwb_datetime,datetime):
#     index=np.asarray([np.where(datetime==cwb_datetime[i])[0][0] for i in range(0,len(cwb_datetime))])
#     return index

#%%
#read 10min operation data
operation_10min=pd.read_csv("D:/database/溫室/伸港Gh/operation/interpolated_data.csv",index_col=0)
#read 10min cwb data
operation_10min.loc[:,'time']=pd.to_datetime(operation_10min.loc[:,'time'])
operation_hourly=operation_10min.groupby(pd.Grouper(key='time',freq='h')).mean().reset_index(drop=False)
operation_datetime=operation_hourly.iloc[:,0].astype(str)
operation_datetime=only_datetime(operation_datetime).astype(int)

#%%
#read cwb data
# cwb_t0=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t0.csv")
cwb_t1=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t1.csv")
cwb_t2=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t2.csv")
cwb_t3=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t3.csv")
cwb_t4=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t4.csv")
cwb_t5=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t5.csv")
cwb_t6=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/timestep_t6.csv")
cwb_datetime=cwb_t1.loc[:,'Report_time']
selected_columns=[5,6,7,8,9,10,11,12]
cwb_combine=(pd.concat([cwb_t1.iloc[:,1:3],cwb_t1.iloc[:,selected_columns],cwb_t2.iloc[:,selected_columns],cwb_t3.iloc[:,selected_columns],cwb_t4.iloc[:,selected_columns],cwb_t5.iloc[:,selected_columns],cwb_t6.iloc[:,selected_columns]],axis=1)).reset_index(drop=True)

#%%
#negative value change to 0
def remove_negative(dataframe):
    dataframe[dataframe<0]=0
    return dataframe
    
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

def interpolate_cwb(df):
    df_array=np.asarray(df).astype(np.float)
    #1st interpolate-inf,0 value
    factor=['dsf','tsf','rh','slp','lwo','psf']
    for i in range(0,len(df_array[0,:])):
        for j in range(0,len(df_array[:,i])):
            if df.columns[i] in factor:
                if df_array[j,i]==0:
                    df_array[j,i]=np.nan

                elif df_array[j,i]==float('+inf'):
                        df_array[j,i]=np.nan                    
                        
            else:
                if df_array[j,i]==float('+inf'):
                    df_array[j,i]=np.nan    
                    
        c_array=df_array[:,i]
        nans, x=nan_helper(c_array)
        c_array[nans]= np.interp(x(nans), x(~nans), c_array[~nans].astype(np.float))    
        df_array[:,i]=c_array
    #2nd interpolate-remove extreme value
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

cwb_combine.iloc[:,2:]=remove_negative(cwb_combine.iloc[:,2:])
clean_data=pd.DataFrame(interpolate_cwb(cwb_combine.iloc[:,2:]))
clean_data.columns=cwb_combine.columns[2:]
cwb_combine=pd.concat([cwb_combine.iloc[:,0:2],clean_data],axis=1)

#%%
#read error indices
outdoor_error_index=pd.read_csv("D:/database/溫室/伸港Gh/outdoor_data/error_index.csv",index_col=0)
indoor_error_index=pd.read_csv("D:/database/溫室/伸港Gh//indoor_data/error_index.csv",index_col=0)
error_index=np.union1d(outdoor_error_index,indoor_error_index)

#%%
#read iot outdoor data
outdoor_complete_hourly=pd.read_csv("D:/database/溫室/伸港Gh/outdoor_data/outdoor_hourly(clean).csv",index_col=0)
outdoor_complete_hourly.iloc[:,1:]=remove_negative(outdoor_complete_hourly.iloc[:,1:])
#remove error data
outdoor_hourly=pd.DataFrame([outdoor_complete_hourly.iloc[index,:] for index in range(0,len(outdoor_complete_hourly)) if index not in error_index])
outdoor_datetime=only_datetime(np.asarray(outdoor_hourly.loc[:,'Time'])).astype(int)
outdoor_datetime=np.asarray([element for index,element in enumerate(outdoor_datetime) if element<=max(cwb_datetime)])

#%%
#read iot indoor data
indoor_complete_hourly=pd.read_csv("D:/database/溫室/伸港Gh/indoor_data/indoor_hourly(clean).csv",index_col=0)
indoor_complete_hourly.iloc[:,1:]=remove_negative(indoor_complete_hourly.iloc[:,1:])
#remove error data
indoor_hourly=pd.DataFrame([indoor_complete_hourly.iloc[index,:] for index in range(0,len(indoor_complete_hourly)) if index not in error_index])
indoor_datetime=only_datetime(np.asarray(indoor_hourly.loc[:,'Time'])).astype(int)
indoor_datetime=np.asarray([element for index,element in enumerate(indoor_datetime) if element<=max(cwb_datetime)])

#%%
#same index
same_element=reduce(np.intersect1d, (operation_datetime,cwb_datetime,outdoor_datetime,indoor_datetime))
same_index_operation=np.squeeze(np.asarray([np.argwhere(operation_datetime==same_element[i]) for i in range(0,len(same_element))]))
same_index_cwb=np.squeeze(np.asarray([np.argwhere(cwb_datetime==same_element[i]) for i in range(0,len(same_element))]))
same_index_outdoor=np.squeeze(np.asarray([np.argwhere(outdoor_datetime==same_element[i]) for i in range(0,len(same_element))]))
same_index_indoor=np.squeeze(np.asarray([np.argwhere(indoor_datetime==same_element[i]) for i in range(0,len(same_element))]))

#%%
#create forecast dataset (t+1 to t+6)
outdoor_output=[];indoor_output=[]
for i in range(1,7):
    if i!=6:
        outdoor_output.append(outdoor_complete_hourly.iloc[i:-(6-i),1:-1].reset_index(drop=True))
        indoor_output.append(indoor_complete_hourly.iloc[i:-(6-i),1:-1].reset_index(drop=True))        
    else:
        outdoor_output.append(outdoor_complete_hourly.iloc[i:,1:-1].reset_index(drop=True))   
        indoor_output.append(indoor_complete_hourly.iloc[i:,1:-1].reset_index(drop=True))   

index_2_remove=pd.DataFrame([i for i in range(len(outdoor_complete_hourly)-6,len(outdoor_complete_hourly))])
outdoor_output=pd.concat(outdoor_output,axis=1)
indoor_output=pd.concat(indoor_output,axis=1)

#%%
writer = pd.ExcelWriter('D:/database/溫室/伸港Gh/same_index.xlsx', engine='xlsxwriter')
#save same index for cwb,outdoor,indoor  
pd.DataFrame(same_index_operation).to_excel(writer,sheet_name="operation_index")
pd.DataFrame(same_index_cwb).to_excel(writer,sheet_name="cwb_index")
pd.DataFrame(same_index_outdoor).to_excel(writer,sheet_name="outdoor_index")
pd.DataFrame(same_index_indoor).to_excel(writer,sheet_name="indoor_index")
writer.save()
cwb_combine.to_csv("D:/database/溫室/伸港Gh/CWB_data/cwb_combine(clean).csv")

indoor_output.to_csv("D:/database/溫室/伸港Gh/indoor_hourly_output(clean).csv")
outdoor_output.to_csv("D:/database/溫室/伸港Gh/outdoor_hourly_output(clean).csv")
index_2_remove.to_csv("D:/database/溫室/伸港Gh/extra_remove_index.csv")