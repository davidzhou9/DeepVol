from __future__ import print_function
from tensorflow.python.training import moving_averages
import logging
import time
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import warnings
warnings.simplefilter("ignore")

TF_DTYPE = tf.float64
MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


class FeedForwardModel(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde, sess):
        self._config = config
        self._bsde = bsde
        self._sess = sess
        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self._extra_train_ops = []
        
    def test(self):
        dw_test, x_test = self._bsde.sample(3)
        #print("spot and vol factor: ", x_train) # use print to output to console
        #print("dw_train: ", dw_train)
        #print("d_W: ", dw_train)
        #logging.info(tf.strings.as_string(x_train))
        loss_Out, init_Out, output_Z_Vals = self._sess.run([self._loss, self._y_init, self._z], feed_dict={self._dw: dw_test,
                                                       self._x: x_test,
                                                       self._is_training: False})
    
        print("Loss out: ", loss_Out)
        print("calc_Grads: ", output_Z_Vals)

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
        self._z_init = tf.Variable(tf.random_uniform([1, self._dim],
                                               minval=-.1, maxval=.1,
                                               dtype=TF_DTYPE))
        
        # tf.ones creates a tensor of all ones (who knew?), tf.shape returns the shape of a tensor (i.e. dimensions in form of tensor)
        # HERE SELF._DW[0] returns NUMBER OF SAMPLES
        # here tf.shape is get dimensioality of the number of samples and appending an extra one
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE) # of samples x 1
        print("all_one_vec: ", all_one_vec)
        
        
        
        #################################### TESTING WITH THE FINAL CONDITION #########################################
        
        """
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
        """
        
        ############################################################################################
        
        y = all_one_vec * self._y_init # number of dimensions by 1
        
        output_z = []
        
        for t in range(self._num_time_interval - 1):
            output_z.append(tf.matmul(all_one_vec, self._z_init))
            
        
        #self._z = tf.matmul(tf.ones(shape = tf.stack([tf.shape(self._dw)[0], self._num_time_interval, self._config.dim]), dtype = TF_DTYPE), self._z_init) # will be time x dimension x dimension
        
        #self._gradient = tf.Variable(tf.zeros([tf.shape(self._dw)[2], 1], dtype = TF_DTYPE)) # dimensionality is number of time steps 
        
        # need to think more critically about how to structure gradient, need it sample, dimensionality, time steps
        gradients = np.zeros((self._num_time_interval, self._dim, self._dim))
        print("gradients: ", gradients)
        print("y: ", y)
        
        print("z output: ", output_z)
        with tf.variable_scope('forward'):
            
            # iterate through time (cooL)
            for t in range(0, self._num_time_interval-1):
                y = y - self._bsde.delta_t * (
                    self._bsde.f_tf(time_stamp[t], self._x[:, :, t], y, output_z[t]) # call f function (e.g. rP)
                ) + tf.reduce_sum(output_z[t] * self._dw[:, :, t], 1, keep_dims=True)
                output_z[t] = self._subnetwork(self._x[:, :, t + 1], str(t + 1)) / self._dim
                #self._z[t, :, :] = self._subnetwork(self._x[:, :, t + 1], str(t + 1)) / self._dim # ALERT: WHY DIVIDE?
                #gradients[t, :, :] = self._z[t + 1, :, :].eval()
            # terminal time
            y = y - self._bsde.delta_t * self._bsde.f_tf(
                time_stamp[-1], self._x[:, :, -2], y, output_z[-1]
            ) + tf.reduce_sum(output_z[-1] * self._dw[:, :, -1], 1, keep_dims=True)
            delta = y - self._bsde.g_tf(self._total_time, self._x[:, :, -1])
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
