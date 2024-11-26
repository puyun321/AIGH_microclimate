# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:42:18 2021

@author: steve
"""


import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
os.chdir("D:/important/work/PM2.5_competition/one_year/new_version(8input)/new_model")
from error_indicator import error_indicator

#%%
"""
Temperature
"""

cnnlstm_temp_forecast=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor/model_performance/cnn-lstm-performance.xlsx', sheet_name='AirTemp_forecast')

cnnlstm_temp_forecast_cwb=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/cnn-lstm-performance(stateless).xlsx', sheet_name='AirTemp_forecast')

real_temp=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/cnn-lstm-performance(stateless).xlsx', sheet_name='AirTemp_realoutput')

plot_index=np.asarray([i for i in range(3850,4150)])
#%%
x=[datetime.strptime(real_temp.iloc[plot_index[i],1],'%Y-%m-%d %H:%M:%S') for i in range(0,len(plot_index))]
plt.plot(x,real_temp.iloc[plot_index,-1],color='black',linewidth=1)
plt.plot(x,cnnlstm_temp_forecast.iloc[plot_index,-1],color='red',linewidth=1)
plt.plot(x,cnnlstm_temp_forecast_cwb.iloc[plot_index,-1],color='green',linewidth=1)

# Format the date into months & days
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=72)) 

cnnlstm_mae=round(error_indicator.np_mae(real_temp.iloc[plot_index,-1],cnnlstm_temp_forecast.iloc[plot_index,-1]),2)
cnnlstm_cwb_mae=round(error_indicator.np_mae(real_temp.iloc[plot_index,-1],cnnlstm_temp_forecast_cwb.iloc[plot_index,-1]),2)

# plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.title('Indoor Temperature')
plt.legend(["observation (MAE)", "C-SL-L (%s)"%cnnlstm_mae,"CL-L (%s)"%cnnlstm_cwb_mae], loc ="lower right", prop={'size': 7,'weight':'bold'})
plt.savefig('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/Indoor Temperature(compare).png')

#%%
"""
Relative humidity
"""

cnnlstm_RH_forecast=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor/model_performance/cnn-lstm-performance.xlsx', sheet_name='RH_forecast')

cnnlstm_RH_forecast_cwb=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/cnn-lstm-performance(stateless).xlsx', sheet_name='RH_forecast')

real_RH=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/cnn-lstm-performance(stateless).xlsx', sheet_name='RH_realoutput')

#%%
x=[datetime.strptime(real_RH.iloc[plot_index[i],1],'%Y-%m-%d %H:%M:%S') for i in range(0,len(plot_index))]
plt.plot(x,real_RH.iloc[plot_index,-1],color='black',linewidth=1)
plt.plot(x,cnnlstm_RH_forecast.iloc[plot_index,-1],color='red',linewidth=1)
plt.plot(x,cnnlstm_RH_forecast_cwb.iloc[plot_index,-1],color='green',linewidth=1)

# Format the date into months & days
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=72)) 

cnnlstm_mae=round(error_indicator.np_mae(real_RH.iloc[plot_index,-1],cnnlstm_RH_forecast.iloc[plot_index,-1]),2)
cnnlstm_cwb_mae=round(error_indicator.np_mae(real_RH.iloc[plot_index,-1],cnnlstm_RH_forecast_cwb.iloc[plot_index,-1]),2)

# plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.title('Indoor Relative Humidity')
plt.legend(["observation (MAE)", "C-SL-L (%s)"%cnnlstm_mae,"CL-L (%s)"%cnnlstm_cwb_mae], loc ="lower right", prop={'size': 7,'weight':'bold'})
plt.savefig('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/Indoor Relative Humidity(compare).png')

#%%
"""
Photosynthetically Active Radiation PAR
"""

cnnlstm_PAR_forecast=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor/model_performance/cnn-lstm-performance.xlsx', sheet_name='PAR_forecast')
store=cnnlstm_PAR_forecast.iloc[:,2:]
store[store<0]=0
cnnlstm_PAR_forecast.iloc[:,2:]=np.asarray(store)
cnnlstm_PAR_forecast_cwb=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/cnn-lstm-performance(stateless).xlsx', sheet_name='PAR_forecast')
store=cnnlstm_PAR_forecast_cwb.iloc[:,2:]
store[store<0]=0
cnnlstm_PAR_forecast_cwb.iloc[:,2:]=np.asarray(store)
real_PAR=pd.read_excel('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/cnn-lstm-performance(stateless).xlsx', sheet_name='PAR_realoutput')

#%%
x=[datetime.strptime(real_PAR.iloc[plot_index[i],1],'%Y-%m-%d %H:%M:%S') for i in range(0,len(plot_index))]
plt.plot(x,real_PAR.iloc[plot_index,-1],color='black',linewidth=1)
plt.plot(x,cnnlstm_PAR_forecast.iloc[plot_index,-1],color='red',linewidth=1)
plt.plot(x,cnnlstm_PAR_forecast_cwb.iloc[plot_index,-1],color='green',linewidth=1)

# Format the date into months & days
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=72)) 

cnnlstm_mae=round(error_indicator.np_mae(real_PAR.iloc[plot_index,-1],cnnlstm_PAR_forecast.iloc[plot_index,-1]),2)
cnnlstm_cwb_mae=round(error_indicator.np_mae(real_PAR.iloc[plot_index,-1],cnnlstm_PAR_forecast_cwb.iloc[plot_index,-1]),2)

# plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.title('Indoor Photosynthetically Active Radiation')
plt.legend(["observation (MAE)", "C-SL-L (%s)"%cnnlstm_mae,"CL-L (%s)"%cnnlstm_cwb_mae], loc ="lower right", prop={'size': 7,'weight':'bold'})
plt.savefig('D:/research/溫室/伸港/hybrid_forecast/physical_model/indoor(CWB)/model_performance/Indoor PAR(compare).png')



