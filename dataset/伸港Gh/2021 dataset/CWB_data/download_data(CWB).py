# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 15:25:37 2021

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
comdText="SELECT * FROM dbo.Grid_Report WHERE Report_time>='2020040100' and GHID='shgb' "
cursor.execute(comdText)
rows = cursor.fetchall()
dataset = np.asarray(rows)
dataset = pd.DataFrame(dataset)

#%%
col_name=['GridLogID','Report_time','GHID','Reportid','TimeID','Predict','Grid_id']
dataset.columns = col_name
sorted_data=dataset.sort_values(by=['Report_time'])
sorted_data=sorted_data.reset_index(drop=True)
sorted_data=sorted_data.iloc[:,1:]
timeid=sorted_data.loc[:,'TimeID']
newtimeid=[]
for i in range(0,len(timeid)):
    newtimeid.append(timeid.iloc[i][0:3])
sorted_data.loc[:,'TimeID']=newtimeid

datetime=np.asarray(sorted_data.loc[:,'Report_time'].drop_duplicates(keep="first")).astype(int)

#%%
# arrange and combine each data
f_0=sorted_data[sorted_data.loc[:,'TimeID']=='f00'].reset_index(drop=True)
f_1=sorted_data[sorted_data.loc[:,'TimeID']=='f01'].reset_index(drop=True)
f_2=sorted_data[sorted_data.loc[:,'TimeID']=='f02'].reset_index(drop=True)
f_3=sorted_data[sorted_data.loc[:,'TimeID']=='f03'].reset_index(drop=True)
f_4=sorted_data[sorted_data.loc[:,'TimeID']=='f04'].reset_index(drop=True)
f_5=sorted_data[sorted_data.loc[:,'TimeID']=='f05'].reset_index(drop=True)
f_6=sorted_data[sorted_data.loc[:,'TimeID']=='f06'].reset_index(drop=True)
different_time_horizon=[f_0,f_1,f_2,f_3,f_4,f_5,f_6]

def seperate_factor(data):
    dsf=data[data.loc[:,'Reportid']=='dsf']
    swi=data[data.loc[:,'Reportid']=='swi']
    tsf=data[data.loc[:,'Reportid']=='tsf']
    vpd=data[data.loc[:,'Reportid']=='vpd']
    slp=data[data.loc[:,'Reportid']=='slp']
    lwo=data[data.loc[:,'Reportid']=='lwo']
    psf=data[data.loc[:,'Reportid']=='psf']
    rh=data[data.loc[:,'Reportid']=='rh']
    
    return dsf,swi,tsf,vpd,slp,lwo,psf,rh

# find the same datetime from different factors
f_0_dsf,f_0_swi,f_0_tsf,f_0_vpd,f_0_slp,f_0_lwo,f_0_psf,f_0_rh=seperate_factor(f_0)
preprocessing_array=[f_0_dsf,f_0_swi,f_0_tsf,f_0_vpd,f_0_slp,f_0_lwo,f_0_psf,f_0_rh]

intersect_datetime=reduce(np.intersect1d,(np.asarray(f_0_dsf.loc[:,'Report_time']),np.asarray(f_0_swi.loc[:,'Report_time']),np.asarray(f_0_tsf.loc[:,'Report_time'])
                ,np.asarray(f_0_vpd.loc[:,'Report_time']),np.asarray(f_0_slp.loc[:,'Report_time']),np.asarray(f_0_lwo.loc[:,'Report_time'])
                ,np.asarray(f_0_psf.loc[:,'Report_time']),np.asarray(f_0_rh.loc[:,'Report_time'])))

def select_specific_datedata(data,intersect_datetime):
    selected_data=[]
    for i in range(0,len(intersect_datetime)):
        selected_data.append(data[data.loc[:,'Report_time']==intersect_datetime[i]])
    selected_data=np.squeeze(np.asarray(selected_data))
    selected_data=pd.DataFrame(selected_data)
    selected_data.columns=data.columns
    return selected_data

#%%
#combine different factors into one array
combine_dataset=[[]*1 for i in range(0,len(different_time_horizon))]
for index,data_timestep in enumerate(different_time_horizon): 
    dsf,swi,tsf,vpd,slp,lwo,psf,rh=seperate_factor(data_timestep)
    preprocessing_array=[dsf,swi,tsf,vpd,slp,lwo,psf,rh]
    for i in range(0,len(preprocessing_array)):
        preprocessing_array[i]=select_specific_datedata(preprocessing_array[i],intersect_datetime)
    concat_element=pd.concat([preprocessing_array[0].iloc[:,0:5],preprocessing_array[1].iloc[:,4],preprocessing_array[2].iloc[:,4],preprocessing_array[3].iloc[:,4],preprocessing_array[4].iloc[:,4],preprocessing_array[5].iloc[:,4],preprocessing_array[6].iloc[:,4],preprocessing_array[7].iloc[:,4]],axis=1)
    concat_element.columns=['Report_time','GHID','Reportid','TimeID','dsf','swi','tsf','vpd','slp','lwo','psf','rh']
    combine_dataset[index]=concat_element

# save each combined timestep dataset into csv
for i in range(0,len(combine_dataset)):
    combine_dataset[i].to_csv("D:/database/溫室/CWB_data/timestep_t%s.csv"%i)
    