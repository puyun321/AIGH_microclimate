# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 22:22:40 2021

@author: steve
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import Model
from keras.engine.input_layer import Input
from keras import backend as K
from keras import regularizers
from keras.models import load_model
from keras.layers import Convolution1D,Dense,concatenate,LSTM,AveragePooling1D
from keras.layers.core import Activation,Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization

os.chdir(os.path.dirname(__file__))
# Move back three levels
os.chdir(os.path.join(os.getcwd(), r'..\..\..'))
target_path = os.path.join(os.getcwd(), r'dataset\伸港Gh\2021 dataset')
os.chdir(target_path)

#%% read data
#read input
cwb_combine=pd.read_csv("CWB_data/cwb_combine(clean).csv",index_col=0)
indoor=pd.read_csv("indoor_data/indoor_10min(clean).csv",index_col=0)
outdoor=pd.read_csv("outdoor_data/outdoor_10min(clean).csv",index_col=0)
operation=pd.read_csv("operation/interpolated_data.csv",index_col=0)

#read error index (if needed)
# extra_remove_index=pd.read_csv("extra_remove_index(10min).csv",index_col=0) 
# error_index=np.asarray(extra_remove_index)

#read same index
same_cwb_index=pd.read_excel('same_index(10min).xlsx',sheet_name="cwb_index",index_col=0)
same_outdoor_index=pd.read_excel('same_index(10min).xlsx',sheet_name="outdoor_index",index_col=0)
same_indoor_index=pd.read_excel('same_index(10min).xlsx',sheet_name="indoor_index",index_col=0)
same_operation_index=pd.read_excel('same_index(10min).xlsx',sheet_name="operation_index",index_col=0)

#read output
indoor_output=pd.read_csv("indoor_10min_output(clean).csv",index_col=0)
outdoor_output=pd.read_csv("outdoor_10min_output(clean).csv",index_col=0)

#%% normalization
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

#%% reshape input data as 2d
# arrange other input as 2d data 
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

# arrange cwb grid data into 2d data
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

# find feature index for each factor in the raw cwb input
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

# arrange operation into 2d data
def operation_convo(dataset,timestep,same_index):
    array=np.asarray(dataset);same_index=np.squeeze(np.asarray(same_index))
    convo_dataset=[[]*1 for i in range(0,len(array)+1)]

    for i in range(int(timestep/6),len(same_index)):
        convo_dataset[i].append(array[same_index[i]-timestep:same_index[i],:])
    convo_dataset=list(filter(None,convo_dataset))
    convo_dataset=np.squeeze(np.asarray(convo_dataset))
    return convo_dataset    

timestep=18

operation_input=operation_convo(operation.iloc[:,1:],timestep,same_operation_index)
cwb_convo_input=norm_cwb.iloc[np.squeeze(np.asarray(same_cwb_index)),:]
cwb_convo_input,factor=cwb_data_arrangement(cwb_convo_input.iloc[int(timestep/6):,:],8)

# indoor_convo_input=convo_input(norm_indoor,timestep,same_indoor_index,error_index)
# outdoor_convo_input=convo_input(norm_outdoor,timestep, same_indoor_index,error_index)

indoor_convo_input=convo_input(norm_indoor,timestep,same_indoor_index)
outdoor_convo_input=convo_input(norm_outdoor,timestep, same_indoor_index)

#%% select factor for indoor, outdoor, and operation data
#select indoor data according factor AirTemp,RH,PAR,SHF,NR,VMF,VWM,VWB
selected_indoor_factor=np.asarray([0,1,2,3,4,5,6,7])
selected_indoor_convo_input=indoor_convo_input[:,:,np.asarray(selected_indoor_factor)]

#select outdoor data according factor OutTemp,OutRH,OutPAR,WindSpeed,WindDir
selected_outdoor_factor=np.asarray([0,1,2])
selected_outdoor_convo_input=outdoor_convo_input[:,:,np.asarray(selected_outdoor_factor)]

#select  operation data according Skykight, Inshade, Northup, Northdown, Southup, Southdown
selected_operation_factor=np.asarray([0,1,2,3,4,5])
selected_operation_convo_input=operation_input[:,:,np.asarray(selected_operation_factor)]/100

#indoor data date
indoor_info=indoor.iloc[np.squeeze(np.asarray(same_indoor_index.iloc[int(timestep/6):,0])),0].reset_index(drop=True)

#%% cnn model output
def model_output(dataset,input_timestep,same_index,feature_num,error_index=0):
    array=np.asarray(dataset);index=np.asarray(error_index);same_index=np.squeeze(np.asarray(same_index))
    convo_dataset=[[]*1 for i in range(0,len(dataset))]
    for i in range((input_timestep-1),len(array)):
        if i in index:
            continue
        else:
            convo_dataset[i-(input_timestep-1)].append(array[i,:])
    convo_dataset=list(filter(None,convo_dataset))
    convo_dataset=[convo_dataset[same_index[j]-(input_timestep-1)] for j in range(int(timestep/6),len(same_index))]
    convo_dataset=np.squeeze((np.asarray(convo_dataset)))
    return convo_dataset

indoor_model_output=model_output(norm_indoor_output,timestep,same_indoor_index,8) #forecast 8 features
outdoor_model_output=model_output(norm_outdoor_output,timestep,same_indoor_index,5) #forecast 5 features

#%% select indoor output data 
# factor: AirTemp,RH,PAR,SHF,NR,VMF,VWM,VWB
output_factor=find_feature_num(indoor_output,8) 
indoor_factor=np.asarray(output_factor)
selected_output_factor=np.asarray([0,1,2])
selected_indoor_output=np.concatenate([indoor_model_output[:,np.asarray(indoor_factor[selected_output_factor,i])] for i in range(0,timestep+1)],axis=1)
denorm_use=np.concatenate([indoor_output.iloc[:,np.asarray(indoor_factor[selected_output_factor,i])] for i in range(0,timestep+1)],axis=1) #outdoor output

#%% split train, validation, test
train_index=[i for i in range(0,int(len(indoor_convo_input)*0.8))]
test_index=[i for i in range(int(len(indoor_convo_input)*0.8),len(indoor_convo_input))]
training_input_one=cwb_convo_input[train_index]
training_input_two=selected_outdoor_convo_input[train_index]
training_input_three=selected_operation_convo_input[train_index]
training_output=selected_indoor_output[train_index]

testing_input_one=cwb_convo_input[test_index]
testing_input_two=selected_outdoor_convo_input[test_index]
testing_input_three=selected_operation_convo_input[test_index]
testing_output=selected_indoor_output[test_index]

#%% construct model
os.chdir(os.path.dirname(__file__))

K.clear_session() 
#cwb data
inputs1 = Input(shape=(6,8))
output=Convolution1D(filters=6,kernel_size=3,padding='same')(inputs1)
output=BatchNormalization()(output)
output=Activation('relu')(output)
output=Convolution1D(filters=12,kernel_size=3,padding='same')(output)
output=BatchNormalization()(output)
output=Activation('relu')(output)
output=Flatten()(output)

#outdoor ioT
inputs2 = Input(shape=(timestep,training_input_two.shape[2]))
output2=LSTM(36,stateful=False,return_sequences=True)(inputs2)
output2=LSTM(36,stateful=False,return_sequences=True)(output2)
output2=LSTM(36,stateful=False,return_sequences=True)(output2)
output2 = AveragePooling1D(pool_size=6,strides=5,padding='valid',data_format='channels_last')(output2)
output2=Flatten()(output2)

#operation data
inputs3 =Input(shape=(timestep,training_input_three.shape[2]))
output3=LSTM(36,kernel_regularizer=regularizers.l2(0.01),stateful=False,return_sequences=True)(inputs3)
output3 = AveragePooling1D(pool_size=6,strides=5,padding='valid',data_format='channels_last')(output3)
output3=Flatten()(output3)

merge_output=concatenate([output,output2,output3],axis=-1)
final_output=Dense(15,activation='relu',kernel_regularizer=regularizers.l2(0.01))(merge_output)
final_output=Dense(len(training_output[0,:]),kernel_regularizer=regularizers.l2(0.01))(final_output)
model = Model(inputs=[inputs1,inputs2,inputs3], outputs=final_output)

learning_rate=1e-4
adam = Adam(learning_rate=learning_rate)
model.compile(optimizer=adam,loss="mse")
earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=0)        
save_path="model/cnn-lstm(stateless).hdf5"
checkpoint =ModelCheckpoint(save_path,save_best_only=True)
callback_list=[earlystopper,checkpoint]        
train_history=model.fit([training_input_one,training_input_two,training_input_three], training_output, epochs=100, batch_size=32,validation_split=0.2,callbacks=callback_list,shuffle=True)

#%% plot training loss and validation loss figure
import matplotlib.pyplot as plt
loss = np.array(train_history.history['loss'])
val_loss = np.array(train_history.history['val_loss'])
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val_loss'])
plt.show()

#%% hybrid model forecasting and denormalization
# forecasting
model=load_model(save_path) 
pred_train_norm=model.predict([training_input_one,training_input_two,training_input_three],batch_size=1)
pred_test_norm=(model.predict([testing_input_one,testing_input_two,testing_input_three],batch_size=1))

# denormalization
def denormalize(df,df2):
    denorm_array = np.asarray(df.copy()); norm =np.asarray(df2.copy())
    for i,_ in enumerate(denorm_array[0]):
        max_value = df[:,i].max()
        min_value = df[:,i].min()
        norm[:,i] = (norm[:,i]* (max_value - min_value))+ min_value
    return norm

pred_train=denormalize(denorm_use,pred_train_norm)
pred_test=denormalize(denorm_use,pred_test_norm)
training_output_denorm=denormalize(denorm_use,training_output)
testing_output_denorm=denormalize(denorm_use,testing_output)

#%% calculate R2 and save into excel
from error_indicator import error_indicator
# total feature number
feature_num=len(selected_output_factor)
# feature index for each indoor factor
pred_factor=find_feature_num(pred_train,feature_num) 
factor_=np.asarray(pred_factor)

training_R2=[[]*1 for i in range(0,feature_num)];testing_R2=[[]*1 for i in range(0,feature_num)]
training_RMSE=[[]*1 for i in range(0,feature_num)];testing_RMSE=[[]*1 for i in range(0,feature_num)]
training_mape=[[]*1 for i in range(0,feature_num)];testing_mape=[[]*1 for i in range(0,feature_num)]
training_mae=[[]*1 for i in range(0,feature_num)];testing_mae=[[]*1 for i in range(0,feature_num)]

# calculate the performance for each factor and each horizon
for j in range(0,feature_num): 
    for i in range(0,timestep+1):
        
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

writer = pd.ExcelWriter('model_performance/cnn-lstm-performance.xlsx', engine='xlsxwriter')
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
for j in range(0,feature_num):
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
writer.close()



        