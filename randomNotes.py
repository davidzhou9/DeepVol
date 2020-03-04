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
"""
z_vals = [[[1, 2], [3, 4], [9, 10]], [[3, 4], [5, 6], [0, 0]], [[7, 8], [9, 10], [0, 0]]]
z_vals_tf = tf.convert_to_tensor(z_vals, dtype = tf.float64)

factor_vals = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] # num sample x num factors x num time interval (fixed a time interval)
factor_vals_tf = tf.convert_to_tensor(factor_vals, dtype = tf.float64)

vol_Arry = tf.math.exp(factor_vals_tf[:, 0] + factor_vals_tf[:, 1])

# z_vals is number of intervals x number of samples x number of factors (Sps we fix a time interval)
print("z_vals_tf: ", z_vals_tf)
print("y_vals_tf: ", y_vals_tf)
print("vol_Arry: ", vol_Arry)

first_Z_Vals = z_vals_tf[0] # num samples x num factors
print("first_Z_Vals: ", first_Z_Vals)

rho_1 = 0.2
rho_2 = 0.3
rho_12 = 0.5
rho_Mat = tf.ones([tf.shape(first_Z_Vals)[0], tf.shape(first_Z_Vals)[1]], dtype = tf.float64) * [rho_1, (rho_2 - rho_1 * rho_12) / (math.sqrt(1 - rho_12**2))]
print("rho_Mat: ", rho_Mat)

Sigma_Grad = tf.reduce_sum(tf.math.multiply(rho_Mat, first_Z_Vals), 1)
print("Sigma_Grad: ", Sigma_Grad)

print("FINAL: ", tf.math.multiply(vol_Arry, Sigma_Grad))

print("Two-norm of z: ", tf.reduce_sum(tf.square(first_Z_Vals), 1, keepdims=False))
"""

num_sample = 4
num_time_interval = 10
T = 0.5
delta_t = T / num_time_interval
sqrt_delta_t = math.sqrt(delta_t)
rho = -0.2

gamma = 0.5
        
r = 0.05
mu_growth = 0.08
v_init = np.ones(1) * 0.0225
reversion_Rate = 10
mean_Rate = 0.0225
vol_Of_Vol = 0.05
        
epsilon = 0.0223 #10**(-7)
maxValue = 0.0227



dw_sample = normal.rvs(size=[num_sample, num_time_interval]) * sqrt_delta_t
x_sample = np.zeros([num_sample, num_time_interval + 1])
x_sample[:, 0] = np.ones(num_sample) * 100
factor = np.exp((r-(0.15**2)/2)*delta_t)
for i in range(num_time_interval):
    x_sample[:, i + 1] = (factor * np.exp(0.15 * dw_sample[:, i])) * x_sample[:, i]
           
print("output: ", np.reshape(dw_sample, (len(dw_sample), 1, num_time_interval)), np.reshape(x_sample, (len(x_sample), 1, num_time_interval + 1)))


"""
dw_sample = normal.rvs(size=[num_sample, num_time_interval]) * sqrt_delta_t
        
v_sample = np.zeros([num_sample, num_time_interval + 1])
v_sample[:, 0] = v_init
   
print(np.ones(len(v_sample[:, 0])) * epsilon)     

for i in range(num_time_interval):
    v_sample[:, i + 1] =  v_sample[:, i] + reversion_Rate * delta_t * (mean_Rate - v_sample[:, i]) + vol_Of_Vol * np.multiply(np.sqrt(v_sample[:, i]), dw_sample[:, i]) 
    v_sample[:, i + 1] = np.maximum(np.minimum(v_sample[:, i + 1], np.ones(len(v_sample[:, 0])) * maxValue), np.ones(len(v_sample[:, 0])) * epsilon)

print("dw_sample: ", dw_sample)            
print("v_sample: ", v_sample)

print("output: ", np.reshape(dw_sample, (len(dw_sample), 1, num_time_interval)), np.reshape(v_sample, (len(v_sample), 1, num_time_interval + 1)))

"""
"""

rho = -0.2949
reversion_Rate = 0.7331
mean_Rate = 0.3407
vol_Of_Vol = 0.7068cle
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
print("x_sample", x_sample)

new_DW = np.zeros(shape = (0, dim, num_time_interval))
new_Process = np.zeros(shape = (0, dim, num_time_interval + 1))

#print("dw_sample: ", dw_sample)

#print("new stuff: ")
for i in range(num_sample):
    currSample = dw_sample[i]
    currXSample = x_sample[i]
    currYSample = y_sample[i]
    
    currOne = currSample[:, 0]
    currTwo = currSample[:, 1]
    print("currSample: ", currSample)
    print("currOne: ", currOne)    


    tempArray = np.ndarray(shape = (dim, num_time_interval), buffer = np.append(currOne, currTwo))
    #print("temp Array: ", tempArray)
    tempArrayOther = np.ndarray(shape = (dim, num_time_interval + 1), buffer = np.append(currXSample, currYSample))
    
    new_DW = np.append(new_DW, np.array([tempArray]), axis = 0)
    new_Process = np.append(new_Process, np.array([tempArrayOther]), axis = 0)
    
print("new_DW: ", new_DW)
print("new_Process: ", new_Process)

#dw_sample.
"""