# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:25:41 2021

@author: steve
"""


import pandas as pd
import numpy as np

import os
import tensorflow as tf
from keras import Model
from keras.engine.input_layer import Input
from keras import backend as K
from keras import regularizers
from keras.models import load_model
from keras.initializers import RandomNormal
from keras.layers import Convolution1D,Dense,concatenate,RepeatVector,LSTM,Lambda,Bidirectional
from keras.layers.core import Activation,Flatten,Reshape
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam_v2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler


#%%
#read input
cwb_combine=pd.read_csv("D:/database/溫室/伸港Gh/CWB_data/cwb_combine(clean).csv",index_col=0)
indoor=pd.read_csv("D:/database/溫室/伸港Gh/indoor_data/indoor_10min(clean).csv",index_col=0)
outdoor=pd.read_csv("D:/database/溫室/伸港Gh/outdoor_data/outdoor_10min(clean).csv",index_col=0)
operation=pd.read_csv("D:/database/溫室/伸港Gh/operation/interpolated_data.csv",index_col=0)

#read error index
extra_remove_index=pd.read_csv("D:/database/溫室/伸港Gh/extra_remove_index(10min).csv",index_col=0) 
error_index=np.asarray(extra_remove_index)
# error_index=np.union1d(np.union1d(outdoor_error_index,indoor_error_index),extra_remove_index)

#read same index
same_cwb_index=pd.read_excel('D:/database/溫室/伸港Gh/same_index(10min).xlsx',sheet_name="cwb_index",index_col=0)
same_outdoor_index=pd.read_excel('D:/database/溫室/伸港Gh/same_index(10min).xlsx',sheet_name="outdoor_index",index_col=0)
same_indoor_index=pd.read_excel('D:/database/溫室/伸港Gh/same_index(10min).xlsx',sheet_name="indoor_index",index_col=0)
same_operation_index=pd.read_excel('D:/database/溫室/伸港Gh/same_index(10min).xlsx',sheet_name="operation_index",index_col=0)

#read output
indoor_output=pd.read_csv("D:/database/溫室/伸港Gh/indoor_10min_output(clean).csv",index_col=0)
outdoor_output=pd.read_csv("D:/database/溫室/伸港Gh/outdoor_10min_output(clean).csv",index_col=0)

#%%
#normalization
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

norm_cwb = normalize(cwb_combine.iloc[:,2:])
norm_indoor = normalize(indoor.iloc[:,1:-1])
norm_outdoor = normalize(outdoor.iloc[:,1:-1])
norm_indoor_output = normalize(indoor_output)
norm_outdoor_output = normalize(outdoor_output)

#%%
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
    convo_dataset=np.squeeze(np.asarray([convo_dataset[same_index[j]-(timestep-1)] for j in range(1,len(same_index))]))
    return convo_dataset

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

    for i in range(1,len(same_index)):
        convo_dataset[i].append(array[same_index[i]-timestep:same_index[i],:])
    convo_dataset=list(filter(None,convo_dataset))
    convo_dataset=np.squeeze(np.asarray(convo_dataset))
    # convo_dataset=np.squeeze(np.asarray(convo_dataset)[same_index[1:]-(timestep-1)])
    return convo_dataset    

operation_input=operation_convo(operation.iloc[:,1:],6,same_operation_index)
cwb_convo_input=norm_cwb.iloc[np.squeeze(np.asarray(same_cwb_index)),:]
cwb_convo_input,factor=cwb_data_arrangement(cwb_convo_input.iloc[1:,:],8)
indoor_convo_input=convo_input(norm_indoor,6,same_indoor_index,error_index)
outdoor_convo_input=convo_input(norm_outdoor,6, same_indoor_index,error_index)

# cwb_convo_input=cwb_combine.iloc[np.squeeze(np.asarray(same_cwb_index)),2:]
# cwb_convo_input,factor=cwb_data_arrangement(cwb_convo_input.iloc[1:,:],8)
# indoor_convo_input=convo_input(indoor.iloc[:,1:],6, same_indoor_index,error_index)
# outdoor_convo_input=convo_input(outdoor.iloc[:,1:],6, same_indoor_index,error_index)

#%%
#select indoor data according factor AirTemp,RH,PAR,SHF,NR,VMF,VWM,VWB
selected_indoor_factor=np.asarray([0,1,2,3,4,5,6,7])
selected_indoor_convo_input=indoor_convo_input[:,:,np.asarray(selected_indoor_factor)]
#select outdoor data according factor OutTemp,OutRH,OutPAR,WindSpeed,WindDir
selected_outdoor_factor=np.asarray([0,1,2])
selected_outdoor_convo_input=outdoor_convo_input[:,:,np.asarray(selected_outdoor_factor)]
#select  operation data according Skykight, Inshade, Northup, Northdown, Southup, Southdown
selected_operation_factor=np.asarray([0,1,2,3,4,5])
selected_operation_convo_input=operation_input[:,:,np.asarray(selected_operation_factor)]/100

#%%
#extract date info
indoor_info=pd.DataFrame([indoor.iloc[index,:] for index in range(0,len(indoor)) if index not in error_index])
indoor_info=indoor_info.iloc[np.squeeze(np.asarray(same_indoor_index.iloc[1:,0])),0].reset_index(drop=True)
indoor_info=indoor_info.iloc[5:].reset_index(drop=True)

#%%

def model_output(dataset,input_timestep,same_index,feature_num,error_index=0):
    array=np.asarray(dataset);index=np.asarray(error_index);same_index=np.squeeze(np.asarray(same_index))
    convo_dataset=[[]*1 for i in range(0,len(dataset))]
    for i in range((input_timestep-1),len(array)):
        if i in index:
            continue
        else:
            convo_dataset[i-(input_timestep-1)].append(array[i,:])
    convo_dataset=list(filter(None,convo_dataset))
    convo_dataset=[convo_dataset[same_index[j]-(input_timestep-1)] for j in range(1,len(same_index))]
    convo_dataset=np.squeeze((np.asarray(convo_dataset)))
    
    return convo_dataset
    
indoor_model_output=model_output(norm_indoor_output,6,same_indoor_index,8,error_index) #forecast 8 features
outdoor_model_output=model_output(norm_outdoor_output,6,same_indoor_index,5,error_index) #forecast 5 features


# indoor_model_output=model_output(indoor_output,6,same_indoor_index,8,error_index) #forecast 8 features
# outdoor_model_output=model_output(outdoor_output,6,same_indoor_index,5,error_index) #forecast 5 features

#%%
#select indoor output data according factor AirTemp,RH,PAR,SHF,NR,VMF,VWM,VWB
output_factor=find_feature_num(indoor_output,8) 
indoor_factor=np.asarray(output_factor)
selected_output_factor=np.asarray([0,1,2,3,4,5,6,7])
selected_indoor_output=np.concatenate([indoor_model_output[:,np.asarray(indoor_factor[selected_output_factor,i])] for i in range(0,7)],axis=1)

#%%
#select outdoor output data according factor AirTemp,RH,PAR,WindSpeed, WindDir
output_factor=find_feature_num(outdoor_output,5) 
outdoor_factor=np.asarray(output_factor)
selected_output_factor=np.asarray([0,1,2])
selected_outdoor_output=np.concatenate([outdoor_model_output[:,np.asarray(outdoor_factor[selected_output_factor,i])] for i in range(0,7)],axis=1)
denorm_use=np.concatenate([outdoor_output.iloc[:,np.asarray(outdoor_factor[selected_output_factor,i])] for i in range(0,7)],axis=1) #outdoor output

#%%
#split train, validation, test
train_index=[i for i in range(0,int(len(indoor_convo_input)*0.8))]
test_index=[i for i in range(int(len(indoor_convo_input)*0.8),len(indoor_convo_input))]
training_input_one=cwb_convo_input[train_index]
training_input_two=selected_outdoor_convo_input[train_index]
training_input_three=selected_operation_convo_input[train_index]
training_output=selected_outdoor_output[train_index]
# training_output=selected_indoor_output[train_index]

testing_input_one=cwb_convo_input[test_index]
testing_input_two=selected_outdoor_convo_input[test_index]
testing_input_three=selected_operation_convo_input[test_index]
testing_output=selected_outdoor_output[test_index]
# testing_output=selected_indoor_output[test_index]

#%%
# os.chdir("D:/research/溫室/伸港/hybrid_forecast/physical_model/model_program")
# from anfis import AnFISLayer
#build model
K.clear_session() 
#cwb
inputs1 = Input(shape=(6,8))
output=LSTM(36,stateful=False,return_sequences=True)(inputs1)
# output=BatchNormalization()(output)
# output=Activation('relu')(output)
output=LSTM(36,stateful=False,return_sequences=True)(output)
# output=BatchNormalization()(output)
# output=Activation('relu')(output)
output=LSTM(36,stateful=False,return_sequences=True)(output)
output=Flatten()(output)

final_output=Dense(15,activation='relu',kernel_regularizer=regularizers.l2(0.01))(output)
final_output=Dense(len(training_output[0,:]),kernel_regularizer=regularizers.l2(0.01))(final_output)
model = Model(inputs=inputs1, outputs=final_output)

learning_rate=1e-4
adam = adam_v2.Adam(learning_rate=learning_rate)
model.compile(optimizer=adam,loss="mse")
earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=0)        
save_path="D:/research/溫室/伸港/hybrid_forecast/physical_model/outdoor(CWB)/model/lstm(stateless).hdf5"
checkpoint =ModelCheckpoint(save_path,save_best_only=True)
callback_list=[earlystopper,checkpoint]        
model.fit(training_input_one, training_output, epochs=100, batch_size=32,validation_split=0.2,callbacks=callback_list,shuffle=True)

#%%
def denormalize(df,df2):
    denorm_array = np.asarray(df.copy()); norm =np.asarray(df2.copy())
    for i,_ in enumerate(denorm_array[0]):
        max_value = df[:,i].max()
        min_value = df[:,i].min()
        norm[:,i] = (norm[:,i]* (max_value - min_value))+ min_value
    return norm

#hybrid model forecasting
# model=load_model(save_path, custom_objects={'AnFISLayer': AnFISLayer}) 
model=load_model(save_path) 
pred_train_norm=model.predict(training_input_one,batch_size=32)
pred_train=denormalize(denorm_use,pred_train_norm)
pred_test_norm=(model.predict(testing_input_one,batch_size=32))
pred_test=denormalize(denorm_use,pred_test_norm)

training_output_denorm=denormalize(denorm_use,training_output)
testing_output_denorm=denormalize(denorm_use,testing_output)

# pred_train=model.predict([training_input_one,training_input_two,training_input_three],batch_size=1)
# pred_test=(model.predict([testing_input_one,testing_input_two,testing_input_three],batch_size=1))

#%%
#calculate R2 and save into excel

os.chdir("D:/important/work/PM2.5_competition/one_year/new_version(8input)/new_model")
from error_indicator import error_indicator
feature_num=len(selected_output_factor)

pred_factor=find_feature_num(pred_train,feature_num) #indoor
factor_=np.asarray(pred_factor)

training_R2=[[]*1 for i in range(0,feature_num)];testing_R2=[[]*1 for i in range(0,feature_num)]
training_RMSE=[[]*1 for i in range(0,feature_num)];testing_RMSE=[[]*1 for i in range(0,feature_num)]
training_mape=[[]*1 for i in range(0,feature_num)];testing_mape=[[]*1 for i in range(0,feature_num)]
training_mae=[[]*1 for i in range(0,feature_num)];testing_mae=[[]*1 for i in range(0,feature_num)]

for j in range(0,feature_num): #number of factors
    for i in range(0,7):
        
        index=factor_[j,i]
        training_R2[j].append(error_indicator.np_R2(training_output_denorm[:,index],pred_train[:,index]))
        testing_R2[j].append(error_indicator.np_R2(testing_output_denorm[:,index],pred_test[:,index]))
        training_RMSE[j].append(error_indicator.np_RMSE(training_output_denorm[:,index],pred_train[:,index]))
        testing_RMSE[j].append(error_indicator.np_RMSE(testing_output_denorm[:,index],pred_test[:,index]))
        training_mape[j].append(error_indicator.np_mape(training_output_denorm[:,index],pred_train[:,index]))
        testing_mape[j].append(error_indicator.np_mape(testing_output_denorm[:,index],pred_test[:,index]))   
        training_mae[j].append(error_indicator.np_mae(training_output_denorm[:,index],pred_train[:,index]))
        testing_mae[j].append(error_indicator.np_mae(testing_output_denorm[:,index],pred_test[:,index]))   
    
training_R2=pd.DataFrame(np.asarray(training_R2))
testing_R2=pd.DataFrame(np.asarray(testing_R2))
training_RMSE=pd.DataFrame(np.asarray(training_RMSE))
testing_RMSE=pd.DataFrame(np.asarray(testing_RMSE)) 
training_mape=pd.DataFrame(np.asarray(training_mape))
testing_mape=pd.DataFrame(np.asarray(testing_mape)) 
training_mae=pd.DataFrame(np.asarray(training_mae))
testing_mae=pd.DataFrame(np.asarray(testing_mae)) 

writer = pd.ExcelWriter('D:/research/溫室/伸港/hybrid_forecast/physical_model/outdoor(CWB)/model_performance/lstm-performance(stateless).xlsx', engine='xlsxwriter')
training_R2.to_excel(writer,sheet_name="training-R2")
testing_R2.to_excel(writer,sheet_name="testing-R2")
training_RMSE.to_excel(writer,sheet_name="training-RMSE")
testing_RMSE.to_excel(writer,sheet_name="testing-RMSE")
training_mape.to_excel(writer,sheet_name="training-mape")
testing_mape.to_excel(writer,sheet_name="testing-mape")
training_mae.to_excel(writer,sheet_name="training-mae")
testing_mae.to_excel(writer,sheet_name="testing-mae")

training_testing_index=[]
for i in range(0,len(indoor_convo_input)):
    if i in train_index:
        training_testing_index.append(0)
    else:
        training_testing_index.append(1)
        
sheet_name=indoor.columns[1:]
sheet_name=sheet_name[selected_output_factor]
for j in range(0,feature_num): #number of factors
    index=factor_[j,:]
    training_testing_index=pd.DataFrame(training_testing_index)
    forecast_result=pd.concat([pd.DataFrame(pred_train[:,index]),pd.DataFrame(pred_test[:,index])]).reset_index(drop=True)
    final_forecast_result=indoor_info
    final_forecast_result=pd.concat([final_forecast_result,training_testing_index,forecast_result],axis=1)
    final_forecast_result.to_excel(writer,sheet_name="%s_forecast"%sheet_name[j])
    model_real_output=pd.concat([pd.DataFrame(training_output_denorm[:,index]),pd.DataFrame(testing_output_denorm[:,index])]).reset_index(drop=True)
    final_real_output=indoor_info
    final_real_output=pd.concat([final_real_output,model_real_output],axis=1)
    final_real_output.to_excel(writer,sheet_name="%s_realoutput"%sheet_name[j])
# Close the Pandas Excel writer and output the Excel file.
writer.save()



        