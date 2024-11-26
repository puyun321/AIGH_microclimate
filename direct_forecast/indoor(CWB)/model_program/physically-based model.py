# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:41:16 2021

@author: steve
"""
import os
import numpy as np
import pandas as pd
import datetime
from math import exp

#%%
os.chdir(os.path.dirname(__file__))
# Move back three levels
os.chdir(os.path.join(os.getcwd(), r'..\..\..'))
os.chdir('direct_forecast\indoor(CWB)\model_program')

#%%

forecast_out_temp=pd.read_excel('model_performance/cnn-lstm-performance(stateless).xlsx',sheet_name='AirTemp_forecast')
forecast_in_PAR=pd.read_excel('model_performance/cnn-lstm-performance(stateless).xlsx',sheet_name='PAR_forecast')
convert=forecast_in_PAR.iloc[:,3:]
convert[convert<0]=0

#%%
# Move back three levels
os.chdir(os.path.join(os.getcwd(), r'..\..\..'))
path = r"dataset\伸港Gh\2021 dataset"
os.chdir(path)

#%%
timestep=18
same_index_operation_=pd.read_excel("same_index(10min).xlsx",sheet_name="operation_index",index_col=0)
same_index_outdoor_=pd.read_excel("same_index(10min).xlsx",sheet_name="outdoor_index",index_col=0)
same_index_indoor_=pd.read_excel("same_index(10min).xlsx",sheet_name="indoor_index",index_col=0)
operation=pd.read_csv("operation/interpolated_data.csv",index_col=0)
outdoor=pd.read_csv("outdoor_data/outdoor_10min(clean).csv",index_col=0)
indoor=pd.read_csv("indoor_data/indoor_10min(clean).csv",index_col=0)
operation=operation.iloc[same_index_operation_.iloc[int(timestep/6):,0],:]
outdoor=outdoor.iloc[same_index_outdoor_.iloc[int(timestep/6):,0],:]
indoor=indoor.iloc[same_index_indoor_.iloc[int(timestep/6):,0],:]
datetime_=np.asarray([str(datetime.datetime.strptime(operation.iloc[i,0],'%Y/%m/%d %H:%M')) for i in range(0,len(operation))])

#%% time factor
tf=[]
for i in range(0,len(datetime_)):
    factor_1=float(datetime_[i][11:13])
    factor_2=float(datetime_[i][14:16])/60
    tf.append(factor_1+factor_2)

#%% define physically-based model formula part 1
# Saturation Vapor Pressure (hPa)
def Tetens(T):
    P = 6.1078*exp(17.27*T/(T+237.3))
    return P

def ArdenBuck(T):
    a = 18.678 - T/234.5
    b = T/(T+257.14)
    P = 6.1121*exp(a*b)
    return P

# Air Density (kg/m^3)
def AirDensity(T, RH, P, method='Tetens'):
    Rd = 287.058
    Rv = 461.495
    
    if(method == 'Tetens'):
        Pvs = 6.1078*exp(17.27*T/(T+237.3))
    elif(method == 'ArdenBuck'):
        a = 18.678 - T/234.5
        b = T/(T+257.14)
        Pvs = 6.1121*exp(a*b)
    else:
        Pvs = 6.1078*exp(17.27*T/(T+237.3))
    
    Pv = Pvs*RH*0.01
    Pd = P - Pv
    Da = Pd*100/(Rd*(T+273.15)) + Pv*100/(Rv*(T+273.15))
    
    return Da

# Air Specific Heat (kJ/kg.K)
def AirCv(T, RH, P, method='Tetens'):
    Cvd = 0.718 #STP
    Cvv = 1.403 #STP
    Rd = 287.058
    Rv = 461.495
    
    if(method == 'Tetens'):
        Pvs = 6.1078*exp(17.27*T/(T+237.3))
    elif(method == 'ArdenBuck'):
        a = 18.678 - T/234.5
        b = T/(T+257.14)
        Pvs = 6.1121*exp(a*b)
    else:
        Pvs = 6.1078*exp(17.27*T/(T+237.3))
    
    Pv = Pvs*RH*0.01
    Pd = P - Pv
    Dv = Pv*100/(Rv*(T+273.15))
    Dd = Pd*100/(Rd*(T+273.15))
    Cv = (Cvd*Dd + Cvv*Dv)/(Dd+Dv)
    
    return Cv

#%% define physically-based model formula part 2

# Variables Setting
data_num=len(operation)
# Cover_area = [312,1560,312]    #[area1, area2, area3, ...]
Cover_area = [1560,187,125,187,125]    
GH_vol = 9360    #unit: m^3
step = 18
duration = 10    #unit: mins

n_par = 0.368
n_photon = 4.56
t_crit = 43.0
s_crit = 1.0
k_ir_1 = 0.3
k_ir_5 = 1.0
k_ir = k_ir_1

h_air = 100    #unit: W/m^2.K
k_cover = 0.5    #unit: W/m.K
l_cover = 0.00015    #unit: m
k_shed = 0.5
l_shed = 0.0002

PAR_GH = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    #[t, t+1, t+2, t+3, t+4, t+5]
Temp_a = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Temp_GH = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    #[t, t+1, t+2, t+3, t+4, t+5, t+6]

# Control Estimation
temp_crit = 27.0
temp_crit_max = 30.0
pred_rslt = []
esti_rslt = []
record_Temp = []
for t in range(0,(len(operation))):

    Shed_table = operation.iloc[t,1:-1]/100     
    
    Temp_GH_esti = []
    Temp_GH_esti_table = []

    Shed_ratio = Shed_table

    TF = tf[t] # time factor
    if t<18:
        Temp = indoor.iloc[t,1]
        Temp_GH[0] = Temp
        k_ir = 0.0081*TF*TF - 0.2154*TF + 1.6984
    else:
        # Temp = indoor.iloc[t,1]
        # Temp_6 = indoor.iloc[t-5,1]
        Temp = esti_rslt[t-1][5] #indoor t
        Temp_6 = esti_rslt[t-1][0] #indoor t-5
        Temp_GH[0] = Temp
        Temp_slope = (Temp-Temp_6 ) / step
        if((Temp >= t_crit) | (Temp_slope >= s_crit)):
            k_ir = k_ir_5
        elif((TF >= 7) & (TF <= 16)):
            k_ir = 0.0081*TF*TF - 0.2154*TF + 1.6984
        else:
            k_ir = k_ir_1
            
    # PAR_output = [outdoor.iloc[t+i,3] for i in range(1,step)] #temporary
    # Temp_a_output = [outdoor.iloc[t+i,1] for i in range(1,step)] #temporary
    PAR_output = [convert.iloc[t,i] for i in range(0,step)] #temporary
    Temp_a_output = [forecast_out_temp.iloc[t,3+i] for i in range(0,step)] 
    
    for i in range(step):
        if(i == 0):
            PAR_GH[i] = PAR_output[i]
            Temp_a[i] = outdoor.iloc[t,1]
        else:
            PAR_GH[i] = PAR_output[i]
            Temp_a[i] = Temp_a_output[i]
    
    cover_column=[0,2,3,4,5]
    Cover_Ir = 0
    for i in range(len(Cover_area)):
        Cover_Ir += Cover_area[i] * (1-Shed_ratio[cover_column[i]])
    
    Cover_U = 0
    for i in range(len(Cover_area)):
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
        Q_loss = (Temp_GH[i]-Temp_a[i]) * Cover_U
        Temp_GH[i+1] = Temp_a[i] + Q_sun/Cover_U
    
        Temp_GH_esti.append(Temp_GH[step])
    Temp_GH_esti_table.append(Temp_GH[1:])
    
    # Efficiency Assessing
    opt_esti = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    #[t, shed_index, Temp_pred, Temp_esti_t+1, t+2, ..., t+6] 
    for i in range(step):
        opt_esti[i] = Temp_GH_esti_table[0][i]
                
    esti_rslt.append(opt_esti)

esti_rslt=np.asarray(esti_rslt)
esti_rslt=pd.DataFrame(esti_rslt)

#%%
real_output=pd.read_csv("indoor_10min_output(clean).csv",index_col=0)
same_indoor_index=pd.read_excel('same_index(10min).xlsx',sheet_name="indoor_index",index_col=0)

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

selected_real_output=model_output(real_output,6,same_indoor_index,8) #forecast 8 features

pred_factor=find_feature_num(selected_real_output,8) #indoor
pred_factor=np.asarray(pred_factor) # all factor indices
temp_factor=pred_factor[0,:] # temperature indices

#%% save estimation performance
# Move back three levels
os.chdir(os.path.join(os.getcwd(), r'..\..\..'))
os.chdir('direct_forecast\indoor(CWB)\model_program')

from error_indicator import error_indicator
train_index=[i for i in range(0,int(len(operation)*0.8))]
test_index=[i for i in range(int(len(operation)*0.8),len(operation))]

training_R2=[[]*1 for i in range(0,timestep)];testing_R2=[[]*1 for i in range(0,timestep)]
training_RMSE=[[]*1 for i in range(0,timestep)];testing_RMSE=[[]*1 for i in range(0,timestep)]
training_mape=[[]*1 for i in range(0,timestep)];testing_mape=[[]*1 for i in range(0,timestep)]
training_mae=[[]*1 for i in range(0,timestep)];testing_mae=[[]*1 for i in range(0,timestep)]

for i in range(0,timestep):
    training_R2[i].append(error_indicator.np_R2(selected_real_output[train_index,temp_factor[i+1]],esti_rslt.iloc[train_index,i]))
    training_RMSE[i].append(error_indicator.np_RMSE(selected_real_output[train_index,temp_factor[i+1]],esti_rslt.iloc[train_index,i]))
    training_mape[i].append(error_indicator.np_mape(selected_real_output[train_index,temp_factor[i+1]],esti_rslt.iloc[train_index,i]))
    training_mae[i].append(error_indicator.np_mae(selected_real_output[train_index,temp_factor[i+1]],esti_rslt.iloc[train_index,i]))
    testing_R2[i].append(error_indicator.np_R2(selected_real_output[test_index,temp_factor[i+1]],esti_rslt.iloc[test_index,i]))
    testing_RMSE[i].append(error_indicator.np_RMSE(selected_real_output[test_index,temp_factor[i+1]],esti_rslt.iloc[test_index,i]))
    testing_mape[i].append(error_indicator.np_mape(selected_real_output[test_index,temp_factor[i+1]],esti_rslt.iloc[test_index,i]))
    testing_mae[i].append(error_indicator.np_mae(selected_real_output[test_index,temp_factor[i+1]],esti_rslt.iloc[test_index,i])) 
    
training_R2=pd.DataFrame(np.asarray(training_R2))
testing_R2=pd.DataFrame(np.asarray(testing_R2))
training_RMSE=pd.DataFrame(np.asarray(training_RMSE))
testing_RMSE=pd.DataFrame(np.asarray(testing_RMSE)) 
training_mape=pd.DataFrame(np.asarray(training_mape))
testing_mape=pd.DataFrame(np.asarray(testing_mape)) 
training_mae=pd.DataFrame(np.asarray(training_mae))
testing_mae=pd.DataFrame(np.asarray(testing_mae)) 
    
writer = pd.ExcelWriter('model_performance/physical_model-performance.xlsx', engine='xlsxwriter')
training_R2.to_excel(writer,sheet_name="training-R2")
testing_R2.to_excel(writer,sheet_name="testing-R2")
training_RMSE.to_excel(writer,sheet_name="training-RMSE")
testing_RMSE.to_excel(writer,sheet_name="testing-RMSE")
training_mape.to_excel(writer,sheet_name="training-mape")
testing_mape.to_excel(writer,sheet_name="testing-mape")
training_mae.to_excel(writer,sheet_name="training-mae")
testing_mae.to_excel(writer,sheet_name="testing-mae")

forecast=pd.concat([pd.DataFrame(datetime_),esti_rslt],axis=1)
forecast.to_excel(writer,sheet_name="temp_forecast")
real=pd.concat([pd.DataFrame(datetime_),pd.DataFrame(selected_real_output[:,temp_factor])],axis=1)
real.to_excel(writer,sheet_name="temp_obs")
# Close the Pandas Excel writer and output the Excel file.
writer.close()

