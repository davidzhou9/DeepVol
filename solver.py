from __future__ import print_function
from tensorflow.python.training import moving_averages
import logging
import time
import numpy as np
import math
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import scipy as sy
import warnings
warnings.simplefilter("ignore")

TF_DTYPE = tf.float64
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


class FeedForwardModel(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde, sess, name):
        self._config = config
        self._bsde = bsde
        self._sess = sess
        self._problem_name = name
        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self._extra_train_ops = []
        self._interest_Rate = bsde.interest_Rate()
        
    def calcPortfolioStrategyHeston(self):
        
        wealth = self._bsde.wealth()
        gamma = self._bsde.gamma()
        eta = self._bsde.eta()
        rho = self._bsde.rho()
        lambdaBar = self._bsde.lambdaBar()
        
        print("wealth: ", wealth)
        print("gamma: ", gamma)
        print("eta: ", eta)
        print("rho: ", rho)
        print("lambdaBar: ", lambdaBar)
        
        
        num_Test_Samples = 3
        
        dw_test, x_test = self._bsde.sample(num_Test_Samples)
        loss_Out, init_Out, output_Z_Vals = self._sess.run([self._loss, self._y_init, self._z], feed_dict={self._dw: dw_test,
                                                       self._x: x_test,
                                                       self._is_training: False})
        print("init out: ", init_Out)
        print("Loss out: ", loss_Out)
        print("Output_Z_Vals: ", output_Z_Vals)
        print("Z_Vals dimensionality: ", output_Z_Vals.shape)
        print("X_test: ", x_test)
        print("X_test type: ", type(x_test))
        reshaped_Z_Vals = []
        ### uucomment ###
        
        print("first element: ", x_test[0][0][0])
        print("first element type: ", type(float(x_test[0][0][0])))
    
        
        print("Test sample: ", 0)
        reshaped_Z_Vals.append(output_Z_Vals[:, 0, :]) # fixed a sample, now time x dimension  x dimension
        current_Sample = output_Z_Vals[:, 0, :]
            
        print("current_Sample: ", current_Sample)
           
        curr_Factors = x_test[0, :, 0] # get x (y, z if exists) at first time step
        curr_V_Factor = x_test[0, 0, 0]
        print("curr_Factors: ", curr_Factors)
        print("curr_V_Factor: ", curr_V_Factor)
        print("current_Sample[0, :]: ", current_Sample[0, :])
           
        mat_Sqrt = self.calculate_Diffusion_Mat(curr_Factors)
        grads = np.linalg.solve(np.transpose(mat_Sqrt), current_Sample[0, :])
        print("mat_Sqrt: ", mat_Sqrt)
        print("grads: ", grads)
            
        g_V = grads[0] # get the delta term
        
        V_x = wealth**(gamma - 1) * math.exp(-init_Out)
        V_xx = (gamma - 1) * wealth**(gamma - 2) * math.exp(-init_Out)
        V_xv = -g_V * wealth**(gamma - 1) * math.exp(-init_Out)
        
        pi_Strategy = -(lambdaBar * V_x + eta * rho * V_xv) / (wealth * V_xx)
        
        print("pi_Strategy: ", pi_Strategy)
        return pi_Strategy
    
    def calcPortfolioStrategyMS(self):
        
        wealth = self._bsde.wealth()
        gamma = self._bsde.gamma()
        mu = self._bsde.mu()
        r = self._bsde.interest_Rate()
        alpha_revert = self._bsde.alpha_Revert()
        delta_revert = self._bsde.delta_Revert()
        rho_1 = self._bsde.rho1()
        rho_2 = self._bsde.rho2()
        nu_f = self._bsde.nu_f()
        nu_s = self._bsde.nu_s()
        
        print("wealth: ", wealth)
        print("gamma: ", gamma)
        print("mu: ", mu)
        print("alpha_revert: ", alpha_revert)
        print("delta_revert: ", delta_revert)
        print("rho1: ", rho_1)
        print("rho2: ", rho_2)
        print("nu_f: ", nu_f)
        print("nu_s: ", nu_s)
        
        
        num_Test_Samples = 3
        
        dw_test, x_test = self._bsde.sample(num_Test_Samples)
        loss_Out, init_Out, output_Z_Vals = self._sess.run([self._loss, self._y_init, self._z], feed_dict={self._dw: dw_test,
                                                       self._x: x_test,
                                                       self._is_training: False})
        print("init out: ", init_Out)
        print("Loss out: ", loss_Out)
        print("Output_Z_Vals: ", output_Z_Vals)
        print("Z_Vals dimensionality: ", output_Z_Vals.shape)
        print("X_test: ", x_test)
        print("X_test type: ", type(x_test))
        reshaped_Z_Vals = []
        ### uucomment ###
        
        print("first element: ", x_test[0][0][0])
        print("first element type: ", type(float(x_test[0][0][0])))
    
        
        print("Test sample: ", 0)
        reshaped_Z_Vals.append(output_Z_Vals[:, 0, :]) # fixed a sample, now time x dimension  x dimension
        current_Sample = output_Z_Vals[:, 0, :]
            
        print("current_Sample: ", current_Sample)
           
        curr_Factors = x_test[0, :, 0] # get x (y, z if exists) at first time step
        curr_Y_Factor = x_test[0, 0, 0]
        curr_Z_Factor = x_test[0, 1, 0]
        print("curr_Factors: ", curr_Factors)
        print("curr_Y_Factor: ", curr_Y_Factor)
        print("curr_Z_Factor: ", curr_Z_Factor)
        
        print("current_Sample[0, :]: ", current_Sample[0, :])
           
        mat_Sqrt = self.calculate_Diffusion_Mat(curr_Factors)
        grads = np.linalg.solve(np.transpose(mat_Sqrt), current_Sample[0, :])
        print("mat_Sqrt: ", mat_Sqrt)
        print("grads: ", grads)
            
        g_y = grads[0]
        g_z = grads[1]
        
        V_x = wealth**(gamma - 1) * math.exp(-init_Out)
        V_xx = (gamma - 1) * wealth**(gamma - 2) * math.exp(-init_Out)
        
        V_xy = -g_y * wealth**(gamma - 1) * math.exp(-init_Out)
        V_xz = -g_z * wealth**(gamma - 1) * math.exp(-init_Out)
        
        pi_Strategy = -((mu - r) * V_x + math.sqrt(2 * alpha_revert) * nu_f * rho_1 * math.exp(curr_Y_Factor + curr_Z_Factor) * V_xy + math.sqrt(2 * delta_revert) * nu_s * rho_2 * math.exp(curr_Y_Factor + curr_Z_Factor) * V_xz) / (math.exp(2 * (curr_Y_Factor + curr_Z_Factor)) * wealth * V_xx)
        
        print("pi_Strategy: ", pi_Strategy)
        return pi_Strategy
    """
    def simPortfolio(self):
        
        wealth = self._bsde.wealth()
        gamma = self._bsde.gamma()
        mu = self._bsde.mu()
        r = self._bsde.interest_Rate()
        alpha_revert = self._bsde.alpha_Revert()
        delta_revert = self._bsde.delta_Revert()
        rho_1 = self._bsde.rho1()
        rho_2 = self._bsde.rho2()
        rho_12 = self._bsde.rho12()
        nu_f = self._bsde.nu_f()
        nu_s = self._bsde.nu_s()
        delta_t = self._bsde.deltaT()
        num_time_interval = self._bsde.numTimeInterval()
        mf = self._bsde.muF()
        ms = self._bsde.muS()
        
        print("wealth: ", wealth)
        print("gamma: ", gamma)
        print("mu: ", mu)
        print("r: ", r)
        print("alpha_revert: ", alpha_revert)
        print("delta_revert: ", delta_revert)
        print("rho1: ", rho_1)
        print("rho2: ", rho_2)
        print("rho12: ", rho_12)
        print("nu_f: ", nu_f)
        print("nu_s: ", nu_s)
        print("delta_t: ", delta_t)
        print("num_time_interval: ", num_time_interval)
        print("mf: ", mf)
        print("ms: ", ms)
        
        num_Samples = 200
        
        dw_Factors, process_Factors, stockPrice = simStockMultiscale(num_Samples, mu, r, alpha_revert, delta_revert, rho_1, rho_2, rho_12, nu_f, nu_s, delta_t, num_time_interval, mf, ms)
        
        loss_Out, init_Out, output_Z_Vals = self._sess.run([self._loss, self._y_init, self._z], feed_dict={self._dw: dw_Factors,
                                                       self._x: process_Factors,
                                                       self._is_training: False})
        print("init out: ", init_Out)
        print("Loss out: ", loss_Out)
        print("Output_Z_Vals: ", output_Z_Vals)
        print("Z_Vals dimensionality: ", output_Z_Vals.shape)
        print("process_Factors: ", process_Factors)
        reshaped_Z_Vals = []
        
        for i in range(num_Samples):
            
            print("Test sample: ", i)
            reshaped_Z_Vals.append(output_Z_Vals[:, i, :]) # fixed a sample, now time x dimension  x dimension
            current_Sample = output_Z_Vals[:, i, :]
            
            print("output_Z_Vals[:, i, :]: ", output_Z_Vals[:, i, :])
           
            curr_Factors = process_Factors[i, :, 0] # get x (y, z if exists) at first time step
            curr_Y_Factor = process_Factors[i, 0, 0]
            curr_Z_Factor = process_Factors[i, 1, 0]
            curr_Stock_Price = stockPrice[i, 0, 0]
            print("curr_Factors: ", curr_Factors)
            print("curr_Stock_Price: ", curr_Stock_Price)
            print("current_Sample[0, :]: ", current_Sample[0, :])
           
            mat_Sqrt = self.calculate_Diffusion_Mat(curr_Factors)
            grads = np.linalg.solve(np.transpose(mat_Sqrt), current_Sample[0, :])
            print("mat_Sqrt: ", mat_Sqrt)
            print("grads: ", grads)
            
            g_y = grads[0]
            g_z = grads[1]
        
            V_x = wealth**(gamma - 1) * math.exp(-init_Out)
            V_xx = (gamma - 1) * wealth**(gamma - 2) * math.exp(-init_Out)
            
            V_xy = -g_y * wealth**(gamma - 1) * math.exp(-init_Out)
            V_xz = -g_z * wealth**(gamma - 1) * math.exp(-init_Out)
        
            nn_pi_Strategy = -((mu - r) * V_x + math.sqrt(2 * alpha_revert) * nu_f * rho_1 * math.exp(curr_Y_Factor + curr_Z_Factor) * V_xy + math.sqrt(2 * delta_revert) * nu_s * rho_2 * math.exp(curr_Y_Factor + curr_Z_Factor) * V_xz) / (math.exp(2 * (curr_Y_Factor + curr_Z_Factor)) * wealth * V_xx)
            merton_Strategy = (mu - r) / (math.exp(2 * (curr_Y_Factor + curr_Z_Factor)) * (1 - gamma))
            
            print("nn_pi_Strategy: ", nn_pi_Strategy)
            print("merton_Strategy: ", merton_Strategy)
            
            # create two portfolios
            nn_Position_In_Stock = (nn_pi_Strategy * wealth) / curr_Stock_Price # gives number of stocks purchased
            nn_Wealth_In_Bonds = wealth - nn_Position_In_Stock * curr_Stock_Price # gives amount of money in risk free
            nn_Portfolio_Value = wealth
            
            merton_Position_In_Stock = (merton_Strategy * wealth) / curr_Stock_Price # gives number of stocks purchased
            merton_Wealth_In_Bonds = wealth - merton_Position_In_Stock * curr_Stock_Price
            merton_Portfolio_Value = wealth
            
            # iterate through all time steps
            print()
            print("------------------------ GOING THRU TIME ------------------------")
            for j in range(1, self._num_time_interval):
               
                # update stock price
                curr_Stock_Price = stockPrice[i, 0, j]
                curr_Factors = process_Factors[i, :, j]
                curr_Y_Factor = process_Factors[i, 0, j]
                curr_Z_Factor = process_Factors[i, 1, j]
                
                print("curr_Stock_Price: ", curr_Stock_Price)
                print("curr_Factors: ", curr_Factors)
                print("curr_Y_Factor: ", curr_Y_Factor)
                print("curr_Z_Factor: ", curr_Z_Factor)
                
                # update NN Portfolio value
                nn_Portfolio_Value = nn_Wealth_In_Bonds * math.exp(r * delta_t) + curr_Stock_Price * nn_Position_In_Stock
                print("nn portfolio after change: ", nn_Portfolio_Value)
                
                # update merton Portfolio value
                merton_Portfolio_Value = merton_Wealth_In_Bonds * math.exp(r * delta_t) + curr_Stock_Price * merton_Position_In_Stock
                print("merton portfolio after change: ", merton_Portfolio_Value)
                   
                mat_Sqrt = self.calculate_Diffusion_Mat(curr_Factors)           
                print("mat_Sqrt: ", mat_Sqrt)
                grads = np.linalg.solve(np.transpose(mat_Sqrt), current_Sample[j, :])
                    
                
            
            
            curr_Stock_Price = x_test[i, 0, -1]
            portfolio = bond_Position * math.exp(self._interest_Rate * delta_T) + curr_Stock_Price * stock_Position
            print()
            print("------------------------ END OF SAMPLE ITER ------------------------")
            print("Portfolio Value: ", portfolio)
            print("Stock Price: ", x_test[i, 0, -1])
            print("PL: ", portfolio - max(x_test[i, 0, -1] - 100, 0))
            
            #settle_Price = self._bsde.g_tf(self._total_time, self._x[i, 0, -1])
            settle_Price = self._x[i, 0, -1] - 100
            print("Settle: ", settle_Price)
            print("PL: ", settle_Price - portfolio)
            running_PL = np.append(running_PL, portfolio - max(x_test[i, 0, -1] - 100, 0))
            
            print("------------------------ END OF SAMPLE ITER ------------------------")
            print("\n")

        print("FINAL AVERAGE P&L: ", np.mean(running_PL))
        print("FINAL STD P&L: ", np.std(running_PL))
        
        return running_PL
        
        
    def simStockMultiscale(self, num_sample, mu, r, alpha_revert, delta_revert, rho_1, rho_2, rho_12, nu_f, nu_s, delta_t, num_time_interval, mf, ms):
        dw_sample = normal.rvs([0, 0, 0], [[delta_t, rho_1 * delta_t, rho_2 * delta_t], 
                                            [rho_1 * delta_t, delta_t, rho_12 * delta_t],
                                            [rho_2 * delta_t, rho_12 * delta_t, delta_t]], size=[num_sample, num_time_interval])
        x_sample = np.zeros([num_sample, num_time_interval + 1])
        x_sample[:, 0] = 100
        
        y_sample = np.zeros([num_sample, num_time_interval + 1])
        y_sample[:, 0] = -1
        
        z_sample = np.zeros([num_sample, num_time_interval + 1])
        z_sample[:, 0] = -1
        
        # validated
        for i in range(num_time_interval):
            vol_factor = np.exp(y_sample[:, i] + z_sample[:, i])
            x_sample[:, i + 1] = x_sample[:, i] * np.exp((r - 0.5 * np.power(vol_factor, 2)) * delta_t + np.multiply(vol_factor, dw_sample[:, i, 0]))
        
            y_sample[:, i + 1] = y_sample[:, i] + alpha_revert * delta_t * (mf - y_sample[:, i]) + nu_f * math.sqrt(2 * alpha_revert) * dw_sample[:, i, 1] 
            z_sample[:, i + 1] = z_sample[:, i] + delta_revert * delta_t * (ms - z_sample[:, i]) + nu_s * math.sqrt(2 * delta) * dw_sample[:, i, 2] 
            
        dw_Factors = np.zeros(shape = (0, 2, num_time_interval))
        process_Factors = np.zeros(shape = (0, 2, num_time_interval + 1))
        
        stockPrice = np.zeros(shape = (0, 1, num_time_interval + 1))
        
        # VALIDATED RESTRUCTURING SCHEME
        for i in range(num_sample):
            currSample = dw_sample[i]
            currXSample = x_sample[i]
            currYSample = y_sample[i]
            currZSample = z_sample[i]
    
            #currOne = currSample[:, 0]
            currTwo = currSample[:, 1]
            currThree = currSample[:, 2]
            ##print("currSample: ", currSample)
            #print("currOne: ", currOne)    

            tempArrayForDWFactors = np.ndarray(shape = (2, num_time_interval), buffer = np.append(currTwo, currThree))
            #print("temp Array: ", tempArray)
            tempArrayForFactors = np.ndarray(shape = (2, num_time_interval + 1), buffer = np.append(currYSample, currZSample))
    
            dw_Factors = np.append(dw_Factors, np.array([tempArrayForDWFactors]), axis = 0)
            process_Factors = np.append(process_Factors, np.array([tempArrayForFactors]), axis = 0)
            
            stockPrice = np.append(stockPrice, currXSample, axis = 0)
            
        return dw_Factors, process_Factors, stockPrice
    """
        
    def delta(self):
        
        num_Test_Samples = 3
        
        dw_test, x_test = self._bsde.sample(num_Test_Samples)
        #print("spot and vol factor: ", x_train) # use print to output to console
        #print("dw_train: ", dw_train)
        #print("d_W: ", dw_train)
        #logging.info(tf.strings.as_string(x_train))
        loss_Out, init_Out, output_Z_Vals = self._sess.run([self._loss, self._y_init, self._z], feed_dict={self._dw: dw_test,
                                                       self._x: x_test,
                                                       self._is_training: False})
        
        # output_Z_Vals is in # of time intervals x number of samples x dimensionality
    
        #output_Z_Vals_Arry = output_Z_Vals.eval()
        print("Loss out: ", loss_Out)
        print("Output_Z_Vals: ", output_Z_Vals)
        print("Z_Vals dimensionality: ", output_Z_Vals.shape)
        print("X_test: ", x_test)
        print("X_test type: ", type(x_test))
        reshaped_Z_Vals = []
        
        #print("output from passing x to f at first time index: ", self._bsde.f_tf(0, x_test[:, :, 1], 0.1, output_Z_Vals[1]).eval())
        #print("output from passing x to g at first time index: ", self._bsde.g_tf(0, x_test[:, :, -1]).eval())
        ### uucomment ###
        
        print("first element: ", x_test[0][0][0])
        print("first element type: ", type(float(x_test[0][0][0])))
    
        
        print("Test sample: ", 0)
        reshaped_Z_Vals.append(output_Z_Vals[:, 0, :]) # fixed a sample, now time x dimension  x dimension
        current_Sample = output_Z_Vals[:, 0, :]
            
        print("output_Z_Vals[:, i, :]: ", output_Z_Vals[:, 0, :])
           
        curr_Factors = x_test[0, :, 0] # get x (y, z if exists) at first time step
        curr_Stock_Price = x_test[0, 0, 0]
        print("curr_Factors: ", curr_Factors)
        print("curr_Stock_Price: ", curr_Stock_Price)
        print("current_Sample[0, :]: ", current_Sample[0, :])
           
        mat_Sqrt = self.calculate_Diffusion_Mat(curr_Factors)
        grads = np.linalg.solve(np.transpose(mat_Sqrt), current_Sample[0, :])
        print("mat_Sqrt: ", mat_Sqrt)
        print("grads: ", grads)
            
        # create replicating portfolio
        stock_Position = grads[0] # get the delta term
        #stock_Position = 0.5
        print("init_Out: ", init_Out)
        print("Stock Position: ", stock_Position)
        # get the first z value
        current_Delta = np.array([])
        current_Delta = np.append(stock_Position, current_Delta)
        return current_Delta
        """
        
    
    def test(self):
        
        num_Test_Samples = 200
        running_PL = np.array([])
        
        dw_test, x_test = self._bsde.sample(num_Test_Samples)
        #print("spot and vol factor: ", x_train) # use print to output to console
        #print("dw_train: ", dw_train)
        #print("d_W: ", dw_train)
        #logging.info(tf.strings.as_string(x_train))
        loss_Out, init_Out, output_Z_Vals = self._sess.run([self._loss, self._y_init, self._z], feed_dict={self._dw: dw_test,
                                                       self._x: x_test,
                                                       self._is_training: False})
        
        # output_Z_Vals is in # of time intervals x number of samples x dimensionality x dimensioinality
    
        #output_Z_Vals_Arry = output_Z_Vals.eval()
        print("Loss out: ", loss_Out)
        print("Output_Z_Vals: ", output_Z_Vals)
        print("Z_Vals dimensionality: ", output_Z_Vals.shape)
        print("X_test: ", x_test)
        print("X_test type: ", type(x_test))
        delta_T = self._total_time / self._num_time_interval
        reshaped_Z_Vals = []
        
        print("first element: ", x_test[0][0][0])
        print("first element type: ", type(float(x_test[0][0][0])))
    
        print("---------------------------- TESTING DELTA HEDGING ----------------------------\n")
        
        # reshape into number of samples x dimensionality x dimensionality
        for i in range(num_Test_Samples):
            
            print("Test sample: ", i)
            reshaped_Z_Vals.append(output_Z_Vals[:, i, :]) # fixed a sample, now time x dimension  x dimension
            current_Sample = output_Z_Vals[:, i, :]
            
            print("output_Z_Vals[:, i, :]: ", output_Z_Vals[:, i, :])
           
            curr_Factors = x_test[i, :, 0] # get x (y, z if exists) at first time step
            curr_Stock_Price = x_test[i, 0, 0]
            print("curr_Factors: ", curr_Factors)
            print("curr_Stock_Price: ", curr_Stock_Price)
            print("current_Sample[0, :]: ", current_Sample[0, :])
           
            mat_Sqrt = self.calculate_Diffusion_Mat(curr_Factors)
            grads = np.linalg.solve(np.transpose(mat_Sqrt), current_Sample[0, :])
            print("mat_Sqrt: ", mat_Sqrt)
            print("grads: ", grads)
            
            # create replicating portfolio
            stock_Position = grads[0] # get the delta term
            #stock_Position = 0.5
            print("init_Out: ", init_Out)
            print("Stock Position: ", stock_Position)
            # get the first z value
           
            bond_Position = init_Out - stock_Position * curr_Stock_Price
            print("Bond position: ", bond_Position)
            portfolio = init_Out
           
            # iterate through all time steps
            print()
            print("------------------------ GOING THRU TIME ------------------------")
            for j in range(1, self._num_time_interval):
               
                # update stock price
                curr_Stock_Price = x_test[i, 0, j]
                curr_Factors = x_test[i, :, j]
                portfolio = bond_Position * math.exp(self._interest_Rate * delta_T) + curr_Stock_Price * stock_Position
                print("portfolio before change: ", portfolio)
                print("curr_Stock_Price: ", curr_Stock_Price)
                print("curr_Factors: ", curr_Factors)
                
                   
                mat_Sqrt = self.calculate_Diffusion_Mat(curr_Factors)           
                print("mat_Sqrt: ", mat_Sqrt)
                grads = np.linalg.solve(np.transpose(mat_Sqrt), current_Sample[j, :])
                    
                print("grads: ", grads)
                stock_Position = grads[0]
                #stock_Position = 0.5
                bond_Position = portfolio - stock_Position * curr_Stock_Price
                print("Stock Position: ", stock_Position)
                print("Bond Position: ", bond_Position)
                print("\n")
            
            
            curr_Stock_Price = x_test[i, 0, -1]
            portfolio = bond_Position * math.exp(self._interest_Rate * delta_T) + curr_Stock_Price * stock_Position
            print()
            print("------------------------ END OF SAMPLE ITER ------------------------")
            print("Portfolio Value: ", portfolio)
            print("Stock Price: ", x_test[i, 0, -1])
            print("PL: ", portfolio - max(x_test[i, 0, -1] - 100, 0))
            
            #settle_Price = self._bsde.g_tf(self._total_time, self._x[i, 0, -1])
            settle_Price = self._x[i, 0, -1] - 100
            print("Settle: ", settle_Price)
            print("PL: ", settle_Price - portfolio)
            running_PL = np.append(running_PL, portfolio - max(x_test[i, 0, -1] - 100, 0))
            
            print("------------------------ END OF SAMPLE ITER ------------------------")
            print("\n")

        print("FINAL AVERAGE P&L: ", np.mean(running_PL))
        print("FINAL STD P&L: ", np.std(running_PL))
        
        return running_PL
        
        #print("reshaped_Z_Vals: ", reshaped_Z_Vals)
        #print("Type of output_Z_Vals: ", type(output_Z_Vals))
        #print("Output_Z_Vals_Arry: ", output_Z_Vals_Arry)        
    """
        
    def calculate_Diffusion_Mat(self, curr_Factors):
        
        # solve for diffusion matrix
        if self._problem_name == 'PricingOptionOneFactor':
            return self._bsde.diffusion_Matrix(curr_Factors[0], max(curr_Factors[1], 0))
        elif self._problem_name == 'PricingOptionNormal':
            return self._bsde.diffusion_Matrix()
        elif self._problem_name == 'PricingOptionMultiFactor':
            return self._bsde.diffusion_Matrix(curr_Factors[0], curr_Factors[1], curr_Factors[2])
        elif self._problem_name == 'HJBHeston':
            return self._bsde.diffusion_Matrix(curr_Factors[0])
        elif self._problem_name == 'HJBMultiscale':
            return self._bsde.diffusion_Matrix(curr_Factors[0], curr_Factors[1])
        #if self._config.get
        #if self._config.get
        return 0
        
    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        
        dw_valid, x_valid = self._bsde.sample(self._config.batch_size) # sample from your bsde problem
        
        
        # can still use batch norm of samples in the validation phase
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}
        
        #feed_dict_valid = {self._x: x_valid, self._is_training: False}
        #self._sess.run(tf.global_variables_initializer())
        # begin sgd iteration
        #for step in range(self._config.num_iterations+1):
        #    if step % self._config.logging_frequency == 0:
        #        loss, init = self._sess.run([self._loss, self._y_init], feed_dict=feed_dict_valid)
        #        elapsed_time = time.time()-start_time+self._t_build
        #        training_history.append([step, loss, init, elapsed_time])
        #        if self._config.verbose:
        #            logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
        #                step, loss, init, elapsed_time))
        #    x_train = self._bsde.sample(self._config.batch_size)
        #    self._sess.run(self._train_ops, feed_dict={self._x: x_train,
        #                                               self._is_training: True})
    
        #print(tf.)
        #return np.array(training_history)
        # initialization
        
        print("num_iterations: ", self._config.num_iterations)
        print("total time intervals: ", self._config.num_time_interval)
        print("# of dimensions: ", self._config.dim)
        self._sess.run(tf.global_variables_initializer())
        # begin sgd iteration
        for step in range(self._config.num_iterations + 1):
           #s print("Iteration step: ", step)
            # logs once every (currently 100) logging freq steps
            if step % self._config.logging_frequency == 0:
                loss, init, z_stack = self._sess.run([self._loss, self._y_init, self._z], feed_dict=feed_dict_valid)
                elapsed_time = time.time()-start_time+self._t_build
                training_history.append([step, loss, init, elapsed_time])
                if self._config.verbose:
                    logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                        step, loss, init, elapsed_time))
                #print("Z stack: ",  z_stack)
                    
            dw_train, x_train = self._bsde.sample(self._config.batch_size)
            #print("spot and vol factor: ", x_train) # use print to output to console
            #print("dw_train: ", dw_train)
            #print("d_W: ", dw_train)
            #logging.info(tf.strings.as_string(x_train))
            self._sess.run(self._train_ops, feed_dict={self._dw: dw_train,
                                                       self._x: x_train,
                                                       self._is_training: True})
    
    
        #print(tf.)
        return np.array(training_history)


    # build function
    # Notes: Called first in the main.
    
    def build(self):
        start_time = time.time()
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t
        
        # placeholders are variables that we will assign data to at a later date
        # will need to feed actual values into these variables at runtime (done in the train function)
        self._dw = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval], name='dW') # TF_DTYPE is tf.float64
        self._x = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval + 1], name='X')
        self._is_training = tf.placeholder(tf.bool)
        
        # initial guess for correct option value (or whatever)
        self._y_init = tf.Variable(tf.random_uniform([1],
                                                     minval=self._config.y_init_range[0],
                                                     maxval=self._config.y_init_range[1],
                                                     dtype=TF_DTYPE)) # y init is picked from a certain range specified in the config file
        
        # guess for the sigma grad term
        #FOR NORMAL BLS / Heston HJB
        """
        self._z_init = tf.Variable(tf.random_uniform([1, self._dim],
                                               minval=0.4, maxval=0.45,
                                               dtype=TF_DTYPE))
        
        
        #FOR HESTON MODEL / Multiscale HJB
        """
        
        lower_Delta = 5.75
        upper_Delta = 5.8
        
        lower_Other = -10.05
        upper_Other = -10
        print("Delta: ", lower_Delta, ", ", upper_Delta)
        print("Other: ", upper_Other, ", ", upper_Other)
        
        self._z_init = tf.Variable(tf.stack([tf.random.uniform([1], lower_Delta, upper_Delta, dtype = tf.float64), 
                                             tf.random.uniform([1], lower_Other, upper_Other, dtype = tf.float64)], axis = 1))
        
        
        # For multiscale model
        
        
        """
        lower_Delta = 5.7
        upper_Delta = 5.75
        
        lower_Other1 = -10.05
        upper_Other1 = -10
        
        
        lower_Other2 = -10.05
        upper_Other2 = -10
        
        
        print("Delta: ", lower_Delta, ", ", upper_Delta)
        print("Other1: ", lower_Other1, ", ", upper_Other1)
        print("Other2: ", lower_Other2, ", ", upper_Other2)
        
        # dimensions 1 by number of z's
        self._z_init = tf.Variable(tf.stack([tf.random.uniform([1], lower_Delta, upper_Delta, dtype = tf.float64), 
                                             tf.random.uniform([1], lower_Other1, upper_Other1, dtype = tf.float64),
                                             tf.random.uniform([1], lower_Other2, upper_Other2, dtype = tf.float64)], axis = 1))
        
        """
        
        # tf.ones creates a tensor of all ones (who knew?), tf.shape returns the shape of a tensor (i.e. dimensions in form of tensor)
        # HERE SELF._DW[0] returns NUMBER OF SAMPLES
        # here tf.shape is get dimensioality of the number of samples and appending an extra one
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE) # of samples x 1
        print("all_one_vec: ", all_one_vec)
        
        y = all_one_vec * self._y_init # number of samples by 1
        
        output_z = [] #time intervals x number of samples x dimensionality
        
        for t in range(self._num_time_interval):
            output_z.append(tf.matmul(all_one_vec, self._z_init)) 
            
        
        #self._z = tf.matmul(tf.ones(shape = tf.stack([tf.shape(self._dw)[0], self._num_time_interval, self._config.dim]), dtype = TF_DTYPE), self._z_init) # will be time x dimension x dimension
        
        #self._gradient = tf.Variable(tf.zeros([tf.shape(self._dw)[2], 1], dtype = TF_DTYPE)) # dimensionality is number of time steps 
        
        # need to think more critically about how to structure gradient, need it sample, dimensionality, time steps
        #gradients = np.zeros((self._num_time_interval, self._dim, self._dim))
        print("y: ", y)
        
        print("z output: ", output_z)
        with tf.variable_scope('forward'):
            #print("----- ITERATING THROUGH TIME STEPS -----")
            # iterate through time (cooL)
            for t in range(0, self._num_time_interval-1):
                #print("current y: ", y)
                #print("current y length: ", tf.shape(y))
                #print("current x: ", self._x)
                #print("current x length: ", tf.shape(self._x))
                
                #print()
                y = y - self._bsde.delta_t * (
                    self._bsde.f_tf(time_stamp[t], self._x[:, :, t], y, output_z[t]) # call f function (e.g. rP)
                ) + tf.reduce_sum(output_z[t] * self._dw[:, :, t], 1, keep_dims=True)
                output_z[t + 1] = self._subnetwork(self._x[:, :, t + 1], str(t + 1)) / self._dim
                
                #self._z[t, :, :] = self._subnetwork(self._x[:, :, t + 1], str(t + 1)) / self._dim # ALERT: WHY DIVIDE?
                #gradients[t, :, :] = self._z[t + 1, :, :].eval()
            # terminal time
            y = y - self._bsde.delta_t * self._bsde.f_tf(
                time_stamp[-1], self._x[:, :, -2], y, output_z[-1]
            ) + tf.reduce_sum(output_z[-1] * self._dw[:, :, -1], 1, keep_dims=True)
            delta = y - self._bsde.g_tf(self._total_time, self._x[:, :, -1])
            #output_z[t] = delta
            #gradients[-1, :, :] = delta.eval()
            # use linear approximation outside the clipped range
            self._loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                                 2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
            
        self._z = tf.stack(output_z)
        # train operations
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self._config.lr_boundaries,
                                                    self._config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time()-start_time
        #return z;
        
        
        # VERIFIED VERSIOn
        """
        start_time = time.time()
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t
        
        # placeholders are variables that we will assign data to at a later date
        # will need to feed actual values into these variables at runtime (done in the train function)
        self._dw = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval], name='dW') # TF_DTYPE is tf.float64
        self._x = tf.placeholder(TF_DTYPE, [None, self._dim, self._num_time_interval + 1], name='X')
        self._is_training = tf.placeholder(tf.bool)
        
        # initial guess for correct option value (or whatever)
        self._y_init = tf.Variable(tf.random_uniform([1],
                                                     minval=self._config.y_init_range[0],
                                                     maxval=self._config.y_init_range[1],
                                                     dtype=TF_DTYPE)) # y init is picked from a certain range specified in the config file
        
        # guess for the sigma grad term
        z_init = tf.Variable(tf.random_uniform([1, self._dim],
                                               minval=-.1, maxval=.1,
                                               dtype=TF_DTYPE))
        
        #################################### TESTING WITH THE FINAL CONDITION #########################################
        
        
        num_Sample = 5
        temp_Sample = normal.rvs([100, 100], [[0.5, -0.7], [-0.7, 0.2]], size=[num_Sample, self._num_time_interval])
        print("temp_Sample: ", temp_Sample)
        
        
        new_Process = np.zeros(shape = (0, self._dim, self._num_time_interval))

        #print("dw_sample: ", dw_sample)

        print("New Process before mods: ", new_Process)
        for i in range(num_Sample):
            
            #for j in range(self._num_time_interval):
                currXSample = temp_Sample[i, :, 0]
                currYSample = temp_Sample[i, :, 1]
                print("currXSample: ", currXSample)
                print("currYSample: ", currYSample)

                tempArrayOther = np.ndarray(shape = (self._dim, self._num_time_interval), buffer = np.append(currXSample, currYSample))
                print("tempArrayOther: ", tempArrayOther)
    
                new_Process = np.append(new_Process, np.array([tempArrayOther]), axis = 0)
    
        print("new_Process: ", new_Process)
        tempTensor = tf.convert_to_tensor(new_Process)
        print("DAVID LOOK: ", self._bsde.g_tf(self._total_time, tempTensor[:, :, -1]).eval())
        
        
        
        print("TEMP 1: ", tempTensor[:, :, -1].eval())
        print("REDUCING: ", tf.reduce_max(tempTensor[:, :, -1], 1, keep_dims=True).eval())
        
        
        ###########################################################################################
        
        # tf.ones creates a tensor of all ones (who knew?), tf.shape returns the shape of a tensor (i.e. dimensions in form of tensor)
        # here tf.shape is get dimensioality of the number of samples and appending an extra one
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
        print("all_one_vec: ", all_one_vec)
        
        y = all_one_vec * self._y_init
        z = tf.matmul(all_one_vec, z_init)
        with tf.variable_scope('forward'):
            
            # iterate through time (cooL)
            for t in range(0, self._num_time_interval-1):
                y = y - self._bsde.delta_t * (
                    self._bsde.f_tf(time_stamp[t], self._x[:, :, t], y, z) # call f function (e.g. rP)
                ) + tf.reduce_sum(z * self._dw[:, :, t], 1, keep_dims=True)
                z = self._subnetwork(self._x[:, :, t + 1], str(t + 1)) / self._dim # ALERT: WHY DIVIDE?
            # terminal time
            y = y - self._bsde.delta_t * self._bsde.f_tf(
                time_stamp[-1], self._x[:, :, -2], y, z
            ) + tf.reduce_sum(z * self._dw[:, :, -1], 1, keep_dims=True)
            delta = y - self._bsde.g_tf(self._total_time, self._x[:, :, -1])
            # use linear approximation outside the clipped range
            self._loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                                 2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        # train operations
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False, dtype=tf.int32)
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    self._config.lr_boundaries,
                                                    self._config.lr_values)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self._loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                                             global_step=global_step, name='train_step')
        all_ops = [apply_op] + self._extra_train_ops
        self._train_ops = tf.group(*all_ops)
        self._t_build = time.time()-start_time
        """


    def _subnetwork(self, x, name):
        with tf.variable_scope(name):
            # standardize the path input first
            # the affine  could be redundant, but helps converge faster
            hiddens = self._batch_norm(x, name='path_input_norm')
            hiddens = x
            for i in range(1, len(self._config.num_hiddens)-1):
                hiddens = self._dense_batch_layer(hiddens,
                                                  self._config.num_hiddens[i],
                                                  activation_fn=tf.nn.relu,
                                                  name='layer_{}'.format(i))
            output = self._dense_batch_layer(hiddens,
                                             self._config.num_hiddens[-1],
                                             activation_fn=None,
                                             name='final_layer')
        return output

    def _dense_batch_layer(self, input_, output_size, activation_fn=None,
                           stddev=5.0, name='linear'):
        with tf.variable_scope(name):
            shape = input_.get_shape().as_list()
            weight = tf.get_variable('Matrix', [shape[1], output_size], TF_DTYPE,
                                     tf.random_normal_initializer(
                                         stddev=stddev/np.sqrt(shape[1]+output_size)))
            hiddens = tf.matmul(input_, weight)
            hiddens_bn = self._batch_norm(hiddens)
        if activation_fn:
            return activation_fn(hiddens_bn)
        else:
            return hiddens_bn

    def _batch_norm(self, x, affine=True, name='batch_norm'):
        """Batch normalization"""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, TF_DTYPE,
                                   initializer=tf.random_normal_initializer(
                                       0.0, stddev=0.1, dtype=TF_DTYPE))
            gamma = tf.get_variable('gamma', params_shape, TF_DTYPE,
                                    initializer=tf.random_uniform_initializer(
                                        0.1, 0.5, dtype=TF_DTYPE))
            moving_mean = tf.get_variable('moving_mean', params_shape, TF_DTYPE,
                                          initializer=tf.constant_initializer(0.0, TF_DTYPE),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, TF_DTYPE,
                                              initializer=tf.constant_initializer(1.0, TF_DTYPE),
                                              trainable=False)
            # These ops will only be preformed when training
            mean, variance = tf.nn.moments(x, [0], name='moments')
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_mean, mean, MOMENTUM))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_variance, variance, MOMENTUM))
            mean, variance = tf.cond(self._is_training,
                                     lambda: (mean, variance),
                                     lambda: (moving_mean, moving_variance))
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, EPSILON)
            y.set_shape(x.get_shape())
            return y
