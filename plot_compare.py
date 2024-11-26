# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:51:44 2024

@author: Steve
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
#%%

temp_forecast=pd.read_excel(r"hybrid_forecast\model_performance\cnn-lstm-performance(cwb).xlsx",sheet_name="AirTemp_forecast")
temp_obs=pd.read_excel(r"hybrid_forecast\model_performance\cnn-lstm-performance(cwb).xlsx",sheet_name="AirTemp_realoutput")

#%%
timestep=1
plt.plot(temp_obs.iloc[:,timestep+3],color="blue")
plt.plot(temp_forecast.iloc[:,timestep+3],color="red")

