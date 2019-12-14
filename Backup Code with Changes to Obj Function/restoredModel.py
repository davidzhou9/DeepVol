# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 23:55:16 2019

@author: david
"""

import tensorflow as tf
import warnings
warnings.simplefilter("ignore")

from config import get_config
from equation import get_equation
from solver import FeedForwardModel
TF_DTYPE = tf.float64
import numpy as np
DELTA_CLIP = 50.0

#extra_train_ops = []
#all_ops = [apply_op] + extra_train_ops
#train_ops = tf.group(*all_ops)

#saver = tf.train.Saver()
    
def importRun(problem_Name, file_Name): 
    #tf.stack([tf.shape(self._dw)[0], 1])
    #print(tf.stack([2, 1]))
    
    num_dims = 2
    num_time_intervals = 10
    
    # initial guess for correct option value (or whatever)
    y_init = tf.Variable(tf.compat.v1.random_uniform([1], minval=0, maxval=5, dtype=TF_DTYPE)) # y init is picked from a certain range specified in the config file
        
    # guess for the sigma grad term
    z_init = tf.Variable(tf.compat.v1.random_uniform([1, num_dims],
                                               minval=-.1, maxval=.1,
                                               dtype=TF_DTYPE))
    all_ones_vec = tf.ones(shape=tf.stack([num_dims, 1]), dtype=TF_DTYPE)
    y = all_ones_vec * y_init
    
    
    z = tf.matmul(tf.matmul(tf.ones(shape = tf.stack([num_time_intervals, num_dims, num_dims]), dtype = TF_DTYPE), all_ones_vec), z_init)
    
    print("HI: ", z[:, :, 0:2])
    print("all_ones_vec: ", all_ones_vec)
    print("y output: ", y)
    print("z output: ", z)
    
    """
    tmpArray = np.zeros((2, 2,))
    tmpArray[:, :] = z.numpy()
    
    print("tmpArray: ", tmpArray)
    """
    """
    with tf.compat.v1.Session() as sess:    
        sess.run(tf.compat.v1.global_variables_initializer())
        
        
        
        saver = tf.compat.v1.train.import_meta_graph(file_Name)
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        
        config = get_config(problem_Name)
        bsde = get_equation(problem_Name, config.dim, config.total_time, config.num_time_interval)
        
        model = FeedForwardModel(config, bsde, sess)
        model.test()
    
        graph = tf.compat.v1.get_default_graph()
        
        
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                                 2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        
        
        dw = tf.compat.v1.placeholder(TF_DTYPE, [None, config.dim, config.num_time_interval], name='dW') # TF_DTYPE is tf.float64
        x = tf.compat.v1.placeholder(TF_DTYPE, [None, config.dim, config.num_time_interval + 1], name='X')
        is_training = tf.compat.v1.placeholder(tf.bool)
        # can still use batch norm of samples in the validation phase
        
        dw_test, x_test = bsde.sample(config.batch_size)
        #print("spot and vol factor: ", x_train) # use print to output to console
        #print("dw_train: ", dw_train)
        #print("d_W: ", dw_train)
        #logging.info(tf.strings.as_string(x_train))
        loss_Out, init_Out = sess.run([loss, y_init], feed_dict={dw: dw_test,
                                                       x: x_train,
                                                       is_training: False})
        
        
        print(P)
        #sess.run(tf.global_variables_initializer())
           #s print("Iteration step: ", step)
            # logs once every (currently 100) logging freq steps
            
            
            if step % self._config.logging_frequency == 0:
                loss, init = self._sess.run([self._loss, self._y_init], feed_dict=feed_dict_valid)
                elapsed_time = time.time()-start_time+self._t_build
                training_history.append([step, loss, init, elapsed_time])
                if self._config.verbose:
                    logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                        step, loss, init, elapsed_time))
            dw_train, x_train = self._bsde.sample(self._config.batch_size)
            #print("spot and vol factor: ", x_train) # use print to output to console
            #print("dw_train: ", dw_train)
            #print("d_W: ", dw_train)
            #logging.info(tf.strings.as_string(x_train))
            self._sess.run(self._train_ops, feed_dict={self._dw: dw_train,
                                                       self._x: x_train,
                                                       self._is_training: True})
    """
    return 0
        
def main():
    problem_Name = "PricingOptionNormal"
    file_Name = "model.ckpt.meta"
    
    
    return importRun(problem_Name, file_Name)
        
if __name__ == '__main__':
    main()