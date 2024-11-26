# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:27:44 2021

@author: steve
"""

import pandas as pd
import numpy as np
import copy
from math import exp, sqrt, log, pi, sin, cos
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
os.chdir("D:/important/work/PM2.5_competition/one_year/new_version(8input)/new_model")

#%%
#read data
Temp_obs_all = pd.read_excel("D:/research/溫室/伸港/hybrid_forecast/model_performance/cnn-lstm-performance(cwb).xlsx",sheet_name = "AirTemp_forecast",index_col=0)
Temp_out_all = pd.read_excel("D:/research/溫室/伸港/direct_forecast/outdoor(CWB)/model_performance/cnn-lstm-performance(stateless).xlsx",sheet_name = "AirTemp_forecast",index_col=0)
PFD_pred_all = pd.read_excel("D:/research/溫室/伸港/hybrid_forecast/model_performance/cnn-lstm-performance(cwb).xlsx",sheet_name = "PAR_forecast",index_col=0)
CO2_pred_all = pd.read_csv("D:/research/溫室/伸港/hybrid_forecast/model_performance/co2_simulate.csv")

#%%
timestep=18
operation = pd.read_csv("D:/database/溫室/伸港Gh/operation/interpolated_data.csv",index_col=0)
operation_set=operation.columns[1:-1]
same_operation_index=pd.read_excel('D:/database/溫室/伸港Gh/same_index(10min).xlsx',sheet_name="operation_index",index_col=0)
def operation_next_timestep(dataset,timestep,same_index):
    array=np.asarray(dataset);same_index=np.squeeze(np.asarray(same_index))
    convo_dataset=[]

    for i in range(int(timestep/6),len(same_index)):
        convo_dataset.append(np.mean(array[same_index[i]+1:same_index[i]+7,:],axis=0))
    convo_dataset=np.asarray(convo_dataset)
    # convo_dataset=np.squeeze(np.asarray(convo_dataset)[same_index[1:]-(timestep-1)])
    return convo_dataset    
operation_t1=operation_next_timestep(operation.iloc[:,1:],timestep,same_operation_index)

def add_missing_columns(dataset):
    add_columns_name=['EastUp','EastDown','WestUp','WestDown']
    add_columns=[]
    for i in range(0,len(dataset)):
        add_columns.append([0]*len(add_columns_name))
    add_columns=np.asarray(add_columns)
    print(np.size(add_columns))
    add_columns=np.concatenate([dataset,add_columns],axis=1)
    return add_columns
operation_t1=add_missing_columns(operation_t1)

#%%
# indoor weather data
Temp_obs = Temp_obs_all.iloc[:,8]
Temp_out = Temp_out_all.iloc[:,8]
PFD_pred = PFD_pred_all.iloc[:,8];PFD_pred[PFD_pred<0]=0
CO2_pred = CO2_pred_all.iloc[:,2]

#%%
# calculate day and hour 
full_datetime = Temp_obs_all.iloc[:,0]
tomato_planttime = ["2020/9/1 00:00","2021/2/28 00:00"]
start_index = (full_datetime[full_datetime==tomato_planttime[0]].index)[0]
end_index = (full_datetime[full_datetime==tomato_planttime[1]].index)[0]

initial_datetime = ["2020/01/01 00:00","2021/01/01 00:00"]

def days_between(d1, d2, dateFormatter):
    d1 = datetime.strptime(d1, dateFormatter)
    d2 = datetime.strptime(d2, dateFormatter)
    return abs((d2 - d1).days)

dateFormatter = "%Y/%m/%d %H:%M"
t_day = []
initial_datetime_=initial_datetime[0]
for i, element in enumerate(full_datetime): 
    day_difference=days_between(element,initial_datetime_,dateFormatter)+1
    if day_difference>365:
        initial_datetime_=initial_datetime[1]        
        day_difference=days_between(element,initial_datetime_,dateFormatter)+1   
    t_day.append(day_difference)

t_hour=[datetime.strptime(element,dateFormatter).hour for i,element in enumerate(full_datetime)]

#%%
# Operation Table
# op_table_ =[[0,0,0,0,0,0,0,0],
#             [0,0,0,0,0,0,0,1],
#             [0,0,0,0,0,0,1,1],
#             [0,0,0,0,1,1,0,0],
#             [1,1,0,0,0,0,0,0],
#             [0,0,1,1,1,1,0,0],
#             [1,1,0,0,0,0,1,1],
#             [0,0,1,1,1,1,1,1],
#             [1,1,1,1,1,1,0,0],
#             [1,1,1,1,1,1,1,1]]
# op_table_=np.asarray(op_table_)

op_table = []
op_table.append([0,0,39,0.20])
op_table.append([1,9.3,38,0.22])
op_table.append([2,18.4,37.5,0.25])
op_table.append([3,22.2,37.3,0.27])
op_table.append([4,37.0,36.4,0.33])
op_table.append([5,44.4,36.0,0.45])
op_table.append([6,55.5,35.4,0.41])
op_table.append([7,63.0,34.9,0.44])
op_table.append([8,81.4,33.9,0.52])
op_table.append([9,100,32.8,0.60])
op_table=np.asarray(op_table)

# op_cont_table = []
# op_cont_table.append("none")
# op_cont_table.append("behind wall")
# op_cont_table.append("front and behind wall")
# op_cont_table.append("only right wall")
# op_cont_table.append("only roof")
# op_cont_table.append("right and left wall")
# op_cont_table.append("roof, front and behind wall")
# op_cont_table.append("all wall")
# op_cont_table.append("roof, right and left wall")
# op_cont_table.append("all")

def find_operation(open_percentage,percentage_table):
    operation_table=copy.deepcopy(percentage_table)
    index2=0
    for index,element in enumerate(operation_table):
        if index==0:
            if open_percentage <= element:
                index2=index
                break
                
            else:
                continue
            
        elif index==len(operation_table)-1:
            index2=index      
            break

        else:
            if open_percentage <= element and open_percentage >= operation_table[index-1]:
                index2=index    
                break
            else:
                continue
    return index2

# def convert_index2operation(index):
#     array=0
#     if index==0:
#         array=[0,0,0,0,0,0,0,0]
#     elif index==1:
#         array=[0,0,0,0,0,0,0,1]
#     elif index==2:
#         array=[0,0,0,0,0,0,1,1]
#     elif index==3:
#         array=[0,0,0,0,1,1,0,0]
#     elif index==4:
#         array=[1,1,0,0,0,0,0,0]
#     elif index==5:
#         array=[0,0,1,1,1,1,0,0]        
#     elif index==6:
#         array=[1,1,0,0,0,0,1,1]  
#     elif index==7:
#         array=[0,0,1,1,1,1,1,1]          
#     elif index==8:
#         array=[1,1,1,1,1,1,0,0]
#     else:          
#         array=[1,1,1,1,1,1,1,1]
        
#     return array

def convert2percentage(array,percentage,array2):
    regr = RandomForestRegressor(max_depth=3, random_state=0)
    regr.fit(array, percentage)
    result=regr.predict(array2)
    return result

# Cumulative TEP
def TEPaccum(Temp, PFD, TEP):
    Temp_opt = 25
    Temp_min = 7
    Temp_max = 48
    
    if(Temp==Temp_opt):
        RTE = 1
    elif((Temp>Temp_min)&(Temp<Temp_opt)):
        RTE = (Temp-Temp_min)/(Temp_opt-Temp_min)
    elif((Temp>Temp_opt)&(Temp<Temp_max)):
        RTE = (Temp_max-Temp)/(Temp_max-Temp_opt)
    else:
        RTE = 0
    
    HRTEP = RTE*PFD*3600*0.000001
    TEP = TEP + HRTEP
    
    return TEP

# Photosynthesis Model
def Pleaf(Temp, PFD, CO2):
    theta = 0.7
    Temp_opt = 25
    Temp_min = 7
    Temp_max = 48
    P_max_opt = 18.90
    
    P_leaf = 0
    if((Temp>=Temp_min)&(Temp<=Temp_max)):
        tau = exp(-3.9489 + 28990/(8.31*(Temp+273)))
        gf = 1000000*0.5*0.21/tau
        Ci = 0.7*CO2 + 0.3*gf
        Qe_CO2 = 6.225*(Ci-gf)/(4*Ci+8*gf)
        Qe = 0.0541*Qe_CO2
        
        C = Temp_opt / Temp
        p1 = (Temp_max-Temp)/(Temp_max-Temp_opt)
        p2 = (Temp-Temp_min)/(Temp_opt-Temp_min)
        p3 = (Temp_opt-Temp_min)/(Temp_max-Temp_opt)
        f_Temp = pow(p1*pow(p2,p3),C)
        P_lmax = P_max_opt*f_Temp
        
        P1 = Qe*PFD + P_lmax
        P2 = pow(Qe*PFD+P_lmax,2)
        P3 = 4*theta*Qe*PFD*P_lmax
        P_leaf = (P1-sqrt(P2-P3)) / (2*theta)
    
    return P_leaf



#%%
Temp_opt_est=[];Pleaf_opt_est=[];opt_est=[];opt_time=[];opt_array=[];open_percentage_est=[]

# for i in range(0,len(Temp_obs)):
for i in range(start_index,end_index+1):

# Optimal Temperature Estimation
    temp_res = 0.1
    temp_range = 20
    temp_min = 7
    temp_max = 48
    

    Temp_Pl_init = max((Temp_obs[i]-temp_range),temp_min)
    Temp_Pl_finl = min((Temp_obs[i]+temp_range), temp_max)
    Temp_Pl_opt = Temp_Pl_init
    Temp_Pl_next = 0
    Pleaf_opt = round(Pleaf(Temp_Pl_opt, PFD_pred[i], CO2_pred[i]),2)
    Pleaf_next = 0
    
    pleaf_list = [[Temp_Pl_opt, Pleaf_opt]]
    pleaf_rslt = [0,0]
    
    for j in range(int((Temp_Pl_finl-Temp_Pl_init)/temp_res)):
        Temp_Pl_next = round((Temp_Pl_init+temp_res*(j+1)),1)
        Pleaf_next = round(Pleaf(Temp_Pl_next, PFD_pred[i], CO2_pred[i]),2)
        pleaf_rslt = [Temp_Pl_next, Pleaf_next]
        pleaf_list.append(pleaf_rslt)
        
        if(Pleaf_next>Pleaf_opt):
            Temp_Pl_opt = Temp_Pl_next
            Pleaf_opt = Pleaf_next
        else:
            continue
        
    # Temperature Control Operation
    tol = 0
    d_temp = Temp_obs[i] - Temp_Pl_opt
    # d_temp_out = Temp_out[i] - Temp_Pc_opt
    op_rslt = [-1,-1]
    
    d_temp_out = abs(Temp_Pl_opt-Temp_out[i])
    
    # if(d_temp<=tol):
    #     d_temp_out = Temp_Pl_opt-Temp_out[i]
        
    # elif(d_temp>0):
    #     d_temp_out = Temp_out[i]-Temp_Pl_opt
    
    open_percentage = (d_temp_out-10.558)*100/(-5.7743)
    operation_index=find_operation(open_percentage,op_table[:,1])
    op_rslt[0] = op_table[operation_index][0]
    op_rslt[1] = abs(d_temp)/op_table[operation_index][3]
    
    Temp_opt_est.append(Temp_Pl_opt)
    Pleaf_opt_est.append(Pleaf_opt)
    opt_time.append(int(op_rslt[1]))
    open_percentage_est.append(op_table[operation_index][1])
    

#%%

writer = pd.ExcelWriter('D:/research/溫室/伸港/operation_forecast/Pleaf/operation_result.xlsx', engine='xlsxwriter')
pd.concat([full_datetime,pd.DataFrame(Temp_opt_est)],axis=1).to_excel(writer,sheet_name="Temp_opt_est")
pd.concat([full_datetime,pd.DataFrame(Pleaf_opt_est)],axis=1).to_excel(writer,sheet_name="Pcanopy_opt_est")
# opt_array=pd.DataFrame(opt_array)
# opt_array.columns=operation_set[0:]
# pd.concat([full_datetime,opt_array],axis=1).to_excel(writer,sheet_name="opt_operation_est_array")
# pd.concat([full_datetime,pd.DataFrame(opt_est)],axis=1).to_excel(writer,sheet_name="opt_operation_est")
pd.concat([full_datetime,pd.DataFrame(opt_time)],axis=1).to_excel(writer,sheet_name="opt_operation_time")
pd.DataFrame(open_percentage_est).to_excel(writer,sheet_name="open_perct_est")

# real_t1=pd.DataFrame(operation_t1[:,:-2])
# real_t1.columns=operation_set[0:8]
# pd.concat([full_datetime,real_t1],axis=1).to_excel(writer,sheet_name="operation_real")
# t1_percentage=convert2percentage(op_table_,op_table[:,1],operation_t1[:,:-2])
# pd.concat([full_datetime,pd.DataFrame(t1_percentage)],axis=1).to_excel(writer,sheet_name="open_perct_real")
writer.save()
    
    