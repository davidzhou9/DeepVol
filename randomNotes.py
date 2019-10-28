# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:12:56 2019

@author: david
"""

"""
USE HESTON
"""

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import math

T = 0.5
num_time_interval = 20
delta_t = T / num_time_interval

num_assets = 1
dim = 2
x_init = np.ones(num_assets) * 100
y_init = np.ones(1) * 0.0989
r = 0.04

"""
Calibration parameters pulled from arxiv article: https://arxiv.org/ftp/arxiv/papers/1502/1502.02963.pdf
"""

rho = -0.2949
reversion_Rate = 0.7331
mean_Rate = 0.3407
vol_Of_Vol = 0.7068
covMatrix = [[delta_t, rho * delta_t], [rho * delta_t, delta_t]]
alpha= 1.0 / dim
num_sample = 3

dw_sample = normal.rvs([0, 0], [[delta_t, rho * delta_t], [rho * delta_t, delta_t]], size=[num_sample, num_time_interval])
x_sample = np.zeros([num_sample, num_time_interval + 1])
x_sample[:, 0] = np.ones(num_sample) * x_init
y_sample = np.zeros([num_sample, num_time_interval + 1])
y_sample[:, 0] = np.ones(num_sample) * y_init
        # for i in xrange(self._n_time):
        # 	x_sample[:, :, i + 1] = (1 + self._mu_bar * self._delta_t) * x_sample[:, :, i] + (
        # 		self._sigma * x_sample[:, :, i] * dw_sample[:, :, i])
#factor = np.exp((r-(sigma**2)/2)*delta_t)
        #+ math.sqrt(y_sample[:, i]) * dw_sample[:, 1, i]
    #* dw_sample[:, 0, i]) * x_sample[:, i] x_sample[:, i + 1] = (factor * np.exp(self._sigma * dw_sample[:, :, i])) * x_sample[:, :, i] # third index is time step

for i in range(num_time_interval):
    y_sample[:, i + 1] = y_sample[:, i] + np.ones(num_sample) * reversion_Rate * mean_Rate * delta_t - reversion_Rate * np.maximum(y_sample[:, i], np.zeros(num_sample)) * delta_t + vol_Of_Vol * np.multiply(np.sqrt(np.maximum(y_sample[:, i], np.zeros(num_sample))), dw_sample[:, i, 1]) 
    x_sample[:, i + 1] = x_sample[:, i] * np.exp((np.ones(num_sample) * r - (np.power(np.maximum(y_sample[:, i + 1], np.zeros(num_sample)), 2)) / 2) * delta_t) * np.exp(np.multiply(np.sqrt(np.maximum(y_sample[:, i + 1], np.zeros(num_sample))), dw_sample[:, i, 0]))
    
    
#print("y_sample", y_sample)
#print("x_sample", x_sample)

new_DW = np.zeros(shape = (0, 2, 20))

print("dw_sample: ", dw_sample)

print("new stuff: ")
for i in range(num_sample):
    currSample = dw_sample[i]
    currOne = currSample[:, 0]
    currTwo = currSample[:, 1]
    print("currSample: ", currSample)
    print("currOne: ", currOne)    

    tempArray = np.ndarray(shape = (2, num_time_interval), buffer = np.append(currOne, currTwo))
    print("temp Array: ", tempArray)
    new_DW = np.append(new_DW, np.array([tempArray]), axis = 0)
    
print("new_DW: ", new_DW)


#dw_sample.
