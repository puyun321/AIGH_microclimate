# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:06:23 2021

@author: steve
"""


import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import random

#%%
indoor=pd.read_csv("D:/database/溫室/伸港Gh/indoor_data/indoor_10min(clean).csv")
outdoor=pd.read_csv("D:/database/溫室/伸港Gh/outdoor_data/outdoor_10min(clean).csv")

#%%

operation=pd.read_csv("D:/database/溫室/伸港Gh/operation/operation_10min(fulldate).csv")
only_data=operation.iloc[:,1:-1].astype("float")

# nan_index=[i for i in range(0,len(only_data)) if np.isnan(only_data.iloc[i,:]).any()]
error_value=[[]*1 for i in range(0,len(only_data))]
for i in range(0,len(only_data)):
    for j in range(0,len(only_data.iloc[i,:])):
        if np.isnan(only_data.iloc[i,j])==True:
            error_value[i].append((j+1))
        if only_data.iloc[i,j]<0:
            error_value[i].append((j+1))                
            
index_5=[i for i in range(0,len(error_value)) if len(error_value[i])==0]
index_4=[i for i in range(0,len(error_value)) if len(error_value[i])==1]
index_3=[i for i in range(0,len(error_value)) if len(error_value[i])==2]
index_2=[i for i in range(0,len(error_value)) if len(error_value[i])==3]

#%%
os.chdir("D:/important/work/PM2.5_competition/one_year/new_version(8input)/new_model")
from error_indicator import error_indicator

#check where index 4 missing
index_4_1=[index_4[i] for i in range(0,len(index_4)) if error_value[index_4[i]][0]==1]
# index_4_2=[index_4[i] for i in range(0,len(index_4)) if error_value[index_4[i]][0]==2]

#%%
#index 4
random_=random.sample(range(len(index_5)), len(index_5))
train_index=np.asarray([random_[i] for i in range(0,int(0.8*len(index_5)))])
test_index=np.asarray(random[len(train_index):])
train_input_0=only_data.iloc[np.asarray(index_5)[train_index],1:]
test_input_0=only_data.iloc[np.asarray(index_5)[test_index],1:]
train_input_1=outdoor.iloc[np.asarray(index_5)[train_index],2:-1]
test_input_1=outdoor.iloc[np.asarray(index_5)[test_index],2:-1]
train_input_2=indoor.iloc[np.asarray(index_5)[train_index],2:-1]
test_input_2=indoor.iloc[np.asarray(index_5)[test_index],2:-1]

train_input=pd.concat([train_input_0,train_input_1,train_input_2],axis=1)
test_input=pd.concat([test_input_0,test_input_1,test_input_2],axis=1).reset_index(drop=True)
train_output=only_data.iloc[np.asarray(index_5)[train_index],0]
test_output=only_data.iloc[np.asarray(index_5)[test_index],0].reset_index(drop=True)

forest = RandomForestRegressor()
forest.fit(train_input, train_output)
train_pred = forest.predict(train_input)
test_pred = forest.predict(test_input)
importances = forest.feature_importances_

train_R2=error_indicator.np_R2(train_output,train_pred);train_RMSE=error_indicator.np_RMSE(train_output,train_pred)
test_R2=error_indicator.np_R2(test_output,test_pred);test_RMSE=error_indicator.np_RMSE(test_output,test_pred)

extra_test_input_0=only_data.iloc[np.asarray(index_4),1:]
extra_test_input_1=outdoor.iloc[np.asarray(index_4),2:-1]
extra_test_input_2=indoor.iloc[np.asarray(index_4),2:-1]
extra_test_input=pd.concat([extra_test_input_0,extra_test_input_1,extra_test_input_2],axis=1).reset_index(drop=True)
data_interpolate=forest.predict(extra_test_input)
only_data.iloc[np.asarray(index_4),0]=data_interpolate

#%%
#check where index 3 missing
# index_3_1=[index_3[i] for i in range(0,len(index_3)) for j in range(0,len(error_value[index_3[i]])) if error_value[index_3[i]][j]==1]
# index_3_2=[index_3[i] for i in range(0,len(index_3)) for j in range(0,len(error_value[index_3[i]])) if error_value[index_3[i]][j]==2]
# index_3_3=[index_3[i] for i in range(0,len(index_3)) for j in range(0,len(error_value[index_3[i]])) if error_value[index_3[i]][j]==3]
# index_3_4=[index_3[i] for i in range(0,len(index_3)) for j in range(0,len(error_value[index_3[i]])) if error_value[index_3[i]][j]==4]
index_3_5=[index_3[i] for i in range(0,len(index_3)) for j in range(0,len(error_value[index_3[i]])) if error_value[index_3[i]][j]==5]
index_3_6=[index_3[i] for i in range(0,len(index_3)) for j in range(0,len(error_value[index_3[i]])) if error_value[index_3[i]][j]==6]

#%%
#index 3
random_=random.sample(range(len(index_5)), len(index_5))
train_index=np.asarray([random_[i] for i in range(0,int(0.8*len(index_5)))])
test_index=np.asarray(random_[len(train_index):])
train_input_0=only_data.iloc[np.asarray(index_5)[train_index],:-2]
test_input_0=only_data.iloc[np.asarray(index_5)[test_index],:-2]
train_input_1=outdoor.iloc[np.asarray(index_5)[train_index],2:-1]
test_input_1=outdoor.iloc[np.asarray(index_5)[test_index],2:-1]
train_input_2=indoor.iloc[np.asarray(index_5)[train_index],2:-1]
test_input_2=indoor.iloc[np.asarray(index_5)[test_index],2:-1]

train_input=pd.concat([train_input_0,train_input_1,train_input_2],axis=1)
test_input=pd.concat([test_input_0,test_input_1,test_input_2],axis=1).reset_index(drop=True)
train_output=only_data.iloc[np.asarray(index_5)[train_index],-2:]
test_output=only_data.iloc[np.asarray(index_5)[test_index],-2:].reset_index(drop=True)

forest = RandomForestRegressor()
forest.fit(train_input, train_output)
train_pred = forest.predict(train_input)
test_pred = forest.predict(test_input)
importances = forest.feature_importances_

train_R2=[];train_RMSE=[];test_R2=[];test_RMSE=[];
for i in range(0,len(train_pred[0])):
    train_R2.append(error_indicator.np_R2(train_output.iloc[:,i],train_pred[:,i]));train_RMSE.append(error_indicator.np_RMSE(train_output.iloc[:,i],train_pred[:,i]))
    test_R2.append(error_indicator.np_R2(test_output.iloc[:,i],test_pred[:,i]));test_RMSE.append(error_indicator.np_RMSE(test_output.iloc[:,i],test_pred[:,i]))

extra_test_input_0=only_data.iloc[np.asarray(index_3),:-2]
extra_test_input_1=outdoor.iloc[np.asarray(index_3),2:-1]
extra_test_input_2=indoor.iloc[np.asarray(index_3),2:-1]
extra_test_input=pd.concat([extra_test_input_0,extra_test_input_1,extra_test_input_2],axis=1).reset_index(drop=True)
data_interpolate=forest.predict(extra_test_input)
only_data.iloc[np.asarray(index_3),-2:]=data_interpolate

#%%
#check where index 2 missing
index_2_1=[index_2[i] for i in range(0,len(index_2)) for j in range(0,len(error_value[index_2[i]])) if error_value[index_2[i]][j]==1]
index_2_2=[index_2[i] for i in range(0,len(index_2)) for j in range(0,len(error_value[index_2[i]])) if error_value[index_2[i]][j]==2]
index_2_3=[index_2[i] for i in range(0,len(index_2)) for j in range(0,len(error_value[index_2[i]])) if error_value[index_2[i]][j]==3]
index_2_4=[index_2[i] for i in range(0,len(index_2)) for j in range(0,len(error_value[index_2[i]])) if error_value[index_2[i]][j]==4]
index_2_5=[index_2[i] for i in range(0,len(index_2)) for j in range(0,len(error_value[index_2[i]])) if error_value[index_2[i]][j]==5]
index_2_6=[index_2[i] for i in range(0,len(index_2)) for j in range(0,len(error_value[index_2[i]])) if error_value[index_2[i]][j]==6]

#%%
#index 2_3
random_=random.sample(range(len(index_5)), len(index_5))
train_index=np.asarray([random_[i] for i in range(0,int(0.8*len(index_5)))])
test_index=np.asarray(random_[len(train_index):])
missing_index=np.asarray([1,3,4])-1;remain_index=np.asarray([2,5,6])-1
train_input_0=only_data.iloc[np.asarray(index_5)[train_index],remain_index]
test_input_0=only_data.iloc[np.asarray(index_5)[test_index],remain_index]
train_input_1=outdoor.iloc[np.asarray(index_5)[train_index],2:-1]
test_input_1=outdoor.iloc[np.asarray(index_5)[test_index],2:-1]
train_input_2=indoor.iloc[np.asarray(index_5)[train_index],2:-1]
test_input_2=indoor.iloc[np.asarray(index_5)[test_index],2:-1]

train_input=pd.concat([train_input_0,train_input_1,train_input_2],axis=1)
test_input=pd.concat([test_input_0,test_input_1,test_input_2],axis=1).reset_index(drop=True)
train_output=only_data.iloc[np.asarray(index_5)[train_index],missing_index]
test_output=only_data.iloc[np.asarray(index_5)[test_index],missing_index].reset_index(drop=True)

forest = RandomForestRegressor()
forest.fit(train_input, train_output)
train_pred = forest.predict(train_input)
test_pred = forest.predict(test_input)
importances = forest.feature_importances_

train_R2=[];train_RMSE=[];test_R2=[];test_RMSE=[];
for i in range(0,len(train_pred[0])):
    train_R2.append(error_indicator.np_R2(train_output.iloc[:,i],train_pred[:,i]));train_RMSE.append(error_indicator.np_RMSE(train_output.iloc[:,i],train_pred[:,i]))
    test_R2.append(error_indicator.np_R2(test_output.iloc[:,i],test_pred[:,i]));test_RMSE.append(error_indicator.np_RMSE(test_output.iloc[:,i],test_pred[:,i]))

extra_test_input_0=only_data.iloc[np.asarray(index_2_3),remain_index]
extra_test_input_1=outdoor.iloc[np.asarray(index_2_3),2:-1]
extra_test_input_2=indoor.iloc[np.asarray(index_2_3),2:-1]
extra_test_input=pd.concat([extra_test_input_0,extra_test_input_1,extra_test_input_2],axis=1).reset_index(drop=True)
data_interpolate=forest.predict(extra_test_input)
only_data.iloc[np.asarray(index_2_3),missing_index]=data_interpolate

#%%
#index 2_5
random_=random.sample(range(len(index_5)), len(index_5))
train_index=np.asarray([random_[i] for i in range(0,int(0.8*len(index_5)))])
test_index=np.asarray(random_[len(train_index):])
missing_index=np.asarray([1,5,6])-1;remain_index=np.asarray([2,3,4])-1
train_input_0=only_data.iloc[np.asarray(index_5)[train_index],remain_index]
test_input_0=only_data.iloc[np.asarray(index_5)[test_index],remain_index]
train_input_1=outdoor.iloc[np.asarray(index_5)[train_index],2:-1]
test_input_1=outdoor.iloc[np.asarray(index_5)[test_index],2:-1]
train_input_2=indoor.iloc[np.asarray(index_5)[train_index],2:-1]
test_input_2=indoor.iloc[np.asarray(index_5)[test_index],2:-1]

train_input=pd.concat([train_input_0,train_input_1,train_input_2],axis=1)
test_input=pd.concat([test_input_0,test_input_1,test_input_2],axis=1).reset_index(drop=True)
train_output=only_data.iloc[np.asarray(index_5)[train_index],missing_index]
test_output=only_data.iloc[np.asarray(index_5)[test_index],missing_index].reset_index(drop=True)

forest = RandomForestRegressor()
forest.fit(train_input, train_output)
train_pred = forest.predict(train_input)
test_pred = forest.predict(test_input)
importances = forest.feature_importances_

train_R2=[];train_RMSE=[];test_R2=[];test_RMSE=[];
for i in range(0,len(train_pred[0])):
    train_R2.append(error_indicator.np_R2(train_output.iloc[:,i],train_pred[:,i]));train_RMSE.append(error_indicator.np_RMSE(train_output.iloc[:,i],train_pred[:,i]))
    test_R2.append(error_indicator.np_R2(test_output.iloc[:,i],test_pred[:,i]));test_RMSE.append(error_indicator.np_RMSE(test_output.iloc[:,i],test_pred[:,i]))

extra_test_input_0=only_data.iloc[np.asarray(index_2_5),remain_index]
extra_test_input_1=outdoor.iloc[np.asarray(index_2_5),2:-1]
extra_test_input_2=indoor.iloc[np.asarray(index_2_5),2:-1]
extra_test_input=pd.concat([extra_test_input_0,extra_test_input_1,extra_test_input_2],axis=1).reset_index(drop=True)
data_interpolate=forest.predict(extra_test_input)
only_data.iloc[np.asarray(index_2_5),missing_index]=data_interpolate

#%%
index_save=np.concatenate((index_5,index_4,index_3,index_2),axis=0)
operation.iloc[index_save,1:-1]=np.asarray(only_data.iloc[index_save,:])
operation.iloc[index_save,:].to_csv("D:/database/溫室/伸港Gh/operation/interpolated_data.csv")

#%%
writer = pd.ExcelWriter('D:/database/溫室/伸港Gh/operation/interpolate_index.xlsx', engine='xlsxwriter')
pd.DataFrame(index_5).to_excel(writer,sheet_name="index_5")
pd.DataFrame(index_4).to_excel(writer,sheet_name="index_4")
pd.DataFrame(index_3).to_excel(writer,sheet_name="index_3")
pd.DataFrame(index_2).to_excel(writer,sheet_name="index_2")

writer.save()