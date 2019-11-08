# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:29:35 2019

@author: david
"""

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import math


num_assets = 1
x_init = np.ones(num_assets) * 100
y_init = np.ones(num_assets) * -1
z_init = np.ones(num_assets) * -1
r = 0.05

total_Time = 0.1
num_time_interval = 50
delta_t = total_Time / num_time_interval
        
# correlation parameters
rho_1 = -0.2
rho_2 = -0.2
rho_12 = 0.0
        
# reversion rate parameters
alpha_revert = 20
delta = 0.1
        
mf = -0.8
ms = -0.8
    
vov_f = 0.5
vov_s = 0.8
        
strike = 95
num_sample = 750000

dw_sample = normal.rvs([0, 0, 0], [[delta_t, rho_1 * delta_t, rho_2 * delta_t], 
                                            [rho_1 * delta_t, delta_t, rho_12 * delta_t],
                                            [rho_2 * delta_t, rho_12 * delta_t, delta_t]], size=[num_sample, num_time_interval])
x_sample = np.zeros([num_sample, num_time_interval + 1])
x_sample[:, 0] = x_init
        
y_sample = np.zeros([num_sample, num_time_interval + 1])
y_sample[:, 0] = y_init
        
z_sample = np.zeros([num_sample, num_time_interval + 1])
z_sample[:, 0] = z_init
        
        
for i in range(num_time_interval):
    vol_factor = np.exp(y_sample[:, i] + z_sample[:, i])
    x_sample[:, i + 1] = x_sample[:, i] * np.exp((r - 0.5 * np.power(vol_factor, 2)) * delta_t + np.multiply(vol_factor, dw_sample[:, i, 0]))
    
    y_sample[:, i + 1] = y_sample[:, i] + alpha_revert * delta_t * (mf - y_sample[:, i]) + vov_f * math.sqrt(2 * alpha_revert) * dw_sample[:, i, 1] 
    z_sample[:, i + 1] = z_sample[:, i] + delta * delta_t * (ms - z_sample[:, i]) + vov_s * math.sqrt(2 * delta) * dw_sample[:, i, 2] 
    #np.ones(num_sample) * self._reversion_Rate * self._mean_Rate * self._delta_t - self._reversion_Rate * np.maximum(y_sample[:, i], np.zeros(num_sample)) * self._delta_t + self._vol_Of_Vol * np.multiply(np.sqrt(np.maximum(y_sample[:, i], np.zeros(num_sample))), dw_sample[:, i, 1]) 
   
#print("Dw_sample: ", dw_sample)
#print("x_sample: ", x_sample)
#print("y_sample: ", y_sample)
#print("z_sample: ", z_sample)

final_Stock_Prices = x_sample[:, num_time_interval]

payoffs = np.maximum(final_Stock_Prices - np.ones(len(final_Stock_Prices)) * strike, np.zeros(len(final_Stock_Prices)))

optionPrice = math.exp(-r * total_Time) * np.mean(payoffs)

stdError = np.std(math.exp(-r * total_Time) * payoffs, ddof = 1) / math.sqrt(num_sample)

print("Strike: ", strike)
print("MC Price: ", optionPrice)
print("Std Error: ", stdError)
    
"""
        new_DW = np.zeros(shape = (0, self._dim, self._num_time_interval))
        new_Process = np.zeros(shape = (0, self._dim, self._num_time_interval + 1))
        
        # VALIDATED RESTRUCTURING SCHEME
        for i in range(num_sample):
            currSample = dw_sample[i]
            currXSample = x_sample[i]
            currYSample = y_sample[i]
            currZSample = z_sample[i]
    
            currOne = currSample[:, 0]
            currTwo = currSample[:, 1]
            currThree = currSample[:, 2]
            ##print("currSample: ", currSample)
            #print("currOne: ", currOne)    

            tempArray = np.ndarray(shape = (self._dim, self._num_time_interval), buffer = np.append(np.append(currOne, currTwo), currThree))
            #print("temp Array: ", tempArray)
            tempArrayOther = np.ndarray(shape = (self._dim, self._num_time_interval + 1), buffer = np.append(np.append(currXSample, currYSample), currZSample))
    
            new_DW = np.append(new_DW, np.array([tempArray]), axis = 0)
            new_Process = np.append(new_Process, np.array([tempArrayOther]), axis = 0)
            
        return new_DW, new_Process
"""