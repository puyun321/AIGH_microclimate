# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:11:27 2019

@author: steve
"""
import numpy as np

class error_indicator():
   
    def np_R(output,output_pred):
        output=np.asarray(output); output_pred=np.asarray(output_pred)
        return np.corrcoef(output.reshape(np.size(output)),output_pred.reshape(np.size(output_pred)), False)[1,0]

    def np_R2(output,output_pred):
        output=np.asarray(output); output_pred=np.asarray(output_pred)
        SS_res = np.sum((output - output_pred)**2)
        SS_tot = np.sum((output - np.mean(output))**2)
        R2 = 1 - (SS_res / SS_tot)
        return R2
        # return np.square(np.corrcoef(output.reshape(np.size(output)),output_pred.reshape(np.size(output_pred)), False)[1,0])

    def np_CE(output,output_pred):
        output=np.asarray(output); output_pred=np.asarray(output_pred)
        return 1-(np.sum(np.square(output-output_pred))/np.sum(np.square(output-np.mean(output))))
    
    def np_RMSE(output,output_pred):
        output=np.asarray(output); output_pred=np.asarray(output_pred)
        rmse=0
        if type(output)==np.float32 or type(output)==float:

            rmse=rmse+np.square(output-output_pred)
            rmse=np.sqrt(rmse)
        else:
            for i in range(0,len(output)):
                rmse=rmse+np.square(output[i]-output_pred[i])
            rmse= np.squeeze(np.sqrt(rmse/len(output)))
        return rmse
        
    def np_mape(output,output_pred):
        output=np.asarray(output); output_pred=np.asarray(output_pred)
        mape=0
        for i in range(0,len(output)):
            if output[i]==0:
                mape=mape+np.abs((output[i]-output_pred[i])/1)
            else:
                mape=mape+np.abs((output[i]-output_pred[i])/output[i])
        mape=mape/len(output)
        return mape

    def np_mb(output,output_pred):
        mb=0
        if type(output)==np.float:
            mb=(output-output_pred)
        else:
            output=np.asarray(output); output_pred=np.asarray(output_pred)
            for i in range(0,len(output)):
                mb=mb+((output[i]-output_pred[i]))
            mb=mb/len(output)
        return mb

    def np_mae(output,output_pred):
        mae=0
        try:
            output=np.asarray(output); output_pred=np.asarray(output_pred)
            for i in range(0,len(output)):
                mae=mae+np.abs((output[i]-output_pred[i]))
            mae=mae/len(output)
        except:
            mae=np.abs(output-output_pred)

        return mae

    def np_error(output,output_pred):
        error=0
        error=output-output_pred
        return error
    
    def kge(observed, simulated):
        r = np.corrcoef(observed, simulated)[0, 1]
        beta = np.mean(simulated) / np.mean(observed)
        gamma = np.std(simulated) / np.std(observed)
    
        kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
        return kge
    
    def calculate_shift(observed, forecasted):
        """
        Calculate the shift or lag between observed and forecasted values in a time series.
        
        Parameters:
        observed (array): An array or list of the observed values in the time series.
        forecasted (array): An array or list of the forecasted values in the time series.
        
        Returns:
        shift (int): The lag or shift between the observed and forecasted values that maximizes the cross-correlation.
        """
    
        # Calculate the cross-correlation between the observed and forecasted values
        xcorr = np.correlate(observed, forecasted, mode='full')
        
        # Find the lag that maximizes the cross-correlation
        shift = np.argmax(xcorr) - (len(observed) - 1)
    
        return shift
