"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).

"""

import json
import logging
import os
import numpy as np
import tensorflow as tf
from config import get_config
from equation import get_equation
from solver import FeedForwardModel
import warnings
warnings.simplefilter("ignore")

TF_DTYPE = tf.float64
FLAGS = tf.compat.v1.flags.FLAGS
#FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('problem_name', 'HJB',
                           """The name of partial differential equation.""")
tf.app.flags.DEFINE_integer('num_run', 1,
                            """The number of experiments to repeatedly run for the same problem.""")
tf.app.flags.DEFINE_string('log_dir1', './logs',
                           """Directory where to write event logs and output array.""")


def main():
    problem_name = FLAGS.problem_name # get probelm name
    config = get_config(problem_name)
    bsde = get_equation(problem_name, config.dim, config.total_time, config.num_time_interval)

    if not os.path.exists(FLAGS.log_dir1): # check to see if this log directory already exists
        os.mkdir(FLAGS.log_dir1)
    path_prefix = os.path.join(FLAGS.log_dir1, problem_name) # create the name of the path file
    
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s')


    ## Order of calls
    # 1) Repeat the experiment the number of times you indicate
    # 2) Create the feed forward network
    # 3) Build the Model
    # 4) Train the model

    # for just pricing
    
    
    vals = np.array([])

    for idx_run in range(1, 51):
        tf.reset_default_graph()
        with tf.Session() as sess:
            #logging.info('Begin to solve %s with run %d' % (problem_name, idx_run))
            
            ###### IMPORTANT ######
            model = FeedForwardModel(config, bsde, sess, problem_name) # create the feed forward model
            if bsde.y_init:
                logging.info('Y0_true: %.4e' % bsde.y_init)
            model.build() # bulid the model
            training_history = model.train() # trin the model
            #if bsde.y_init:
            #    logging.info('relative error of Y0: %s',
            #                 '{:.2%}'.format(
            #                     abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
            # save training history
            
        vals = np.append(vals, training_history[len(training_history) - 1, 2])
        
    vals = np.append(vals, bsde._total_time)
    
    num_Iter = 1
    while os.path.exists('{}_training_history_{}_{}.csv'.format(path_prefix, idx_run, num_Iter)):
        num_Iter += 1
            
    np.savetxt('{}_training_history_{}_{}.csv'.format(path_prefix, idx_run, num_Iter),
                       vals,
                       fmt=['%.5e'],
                       delimiter=",",
                       header="target_value",
                       comments='')
    
    
    
    """
    for idx_run in range(1, 2):
        tf.reset_default_graph()
        with tf.Session() as sess:
            #logging.info('Begin to solve %s with run %d' % (problem_name, idx_run))
            
            ###### IMPORTANT ######
            model = FeedForwardModel(config, bsde, sess, problem_name) # create the feed forward model
            if bsde.y_init:
                logging.info('Y0_true: %.4e' % bsde.y_init)
            model.build() # bulid the model
            training_history = model.train() # trin the model
            #if bsde.y_init:
            #    logging.info('relative error of Y0: %s',
            #                 '{:.2%}'.format(
            #                     abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))
            # save training history
            
            
            num_Iter = 1
            while os.path.exists('{}_training_history_FORPL_{}_{}.csv'.format(path_prefix, idx_run, num_Iter)):
                num_Iter += 1
            
            #training_history = np.append(training_history, bsde._total_time)
            np.savetxt('{}_training_history_FORPL_{}_{}.csv'.format(path_prefix, idx_run, num_Iter),
                       training_history,
                       fmt=['%d', '%.5e', '%.5e', '%d'],
                       delimiter=",",
                       header="step,loss_function,target_value,elapsed_time",
                       comments='')
        
            output = model.test()
            
            num_Iter = 1
            while os.path.exists('{}_PLfigures_{}.csv'.format(path_prefix, num_Iter)):
                num_Iter += 1
                
            output = np.append(output, bsde._num_time_interval)
            np.savetxt('{}_PLfigures_{}.csv'.format(path_prefix, num_Iter),
                       output,
                       fmt=['%f'],
                       delimiter=",",
                       header="PL",
                       comments='')
    """ 

if __name__ == '__main__':
    main()
