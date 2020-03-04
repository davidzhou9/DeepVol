import numpy as np


class Config(object):
    n_layer = 2
    batch_size = 5000 # original was 5000
    valid_size = 256 # original was 256
    step_boundaries = [2000, 4000]
    num_iterations = 2
    logging_frequency = 10 # original is 100
    verbose = True
    y_init_range = [0, 1]


class AllenCahnConfig(Config):
    total_time = 0.3
    num_time_interval = 20
    dim = 100
    lr_values = list(np.array([5e-4, 5e-4]))
    lr_boundaries = [2000]
    num_iterations = 4000
    num_hiddens = [dim, dim + 10, dim + 10, dim]
    y_init_range = [0.3, 0.6]


class HJBConfig(Config):
    # Y_0 is about 4.5901.
    dim = 100
    total_time = 1.0
    num_time_interval = 20
    lr_boundaries = [400]
    num_iterations = 2000
    lr_values = list(np.array([1e-2, 1e-2]))
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [0, 1]


class PricingOptionConfig(Config):
    dim = 2
    total_time = 0.5
    num_time_interval = 100
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_iterations = 4000
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [0, 1]
    
class PricingOptionNormalConfig(Config):
    dim = 1
    total_time = 0.1
    num_time_interval = 80
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_iterations = 150
    num_hiddens = [dim, 5, 5, dim]
    y_init_range = [3.2, 3.3]
    

class PricingOptionOneFactorConfig(Config):
    dim = 2
    total_time = 0.2
    num_time_interval = 80
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_iterations = 150 # original was 5000
    num_hiddens = [dim, 5, 5, dim]
    y_init_range = [0.3, 0.4]
    
class PricingOptionMultiFactorConfig(Config):
    dim = 3
    total_time = 0.2
    num_time_interval = 80
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_iterations = 150
    num_hiddens = [dim, 5, 5, dim]
    y_init_range = [0.4, 0.5]
    
class HJBHestonConfig(Config):
    # Y_0 is about 4.5901.
    dim = 1
    total_time = 0.2
    num_time_interval = 80  #CHECK THIS DAVID
    lr_boundaries = [400]
    num_iterations = 150
    lr_values = list(np.array([1e-2, 1e-2]))
    num_hiddens = [dim, 5, 5, dim]
    y_init_range = [-0.2, -0.1]    
    
class HJBMultiscaleConfig(Config):
    # Y_0 is about 4.5901.
    dim = 2
    total_time = 0.2
    num_time_interval = 80  #CHECK THIS DAVID
    lr_boundaries = [400]
    num_iterations = 3000
    lr_values = list(np.array([1e-2, 1e-2]))
    num_hiddens = [dim, 5, 5, dim]
    y_init_range = [-0.2, -0.1]

class PricingDefaultRiskConfig(Config):
    dim = 100
    total_time = 1
    num_time_interval = 40
    lr_values = list(np.array([8e-3, 8e-3]))
    lr_boundaries = [3000]
    num_iterations = 6000
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [40, 50]


class BurgesTypeConfig(Config):
    dim = 50
    total_time = 0.2
    num_time_interval = 30
    lr_values = list(np.array([1e-2, 1e-3, 1e-4]))
    lr_boundaries = [15000, 25000]
    num_iterations = 30000
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [2, 4]


class QuadraticGradientsConfig(Config):
    dim = 100
    total_time = 1.0
    num_time_interval = 30
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_iterations = 4000
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [2, 4]


class ReactionDiffusionConfig(Config):
    dim = 100
    total_time = 1.0
    num_time_interval = 30
    lr_values = list(np.array([1e-2, 1e-2, 1e-2]))
    lr_boundaries = [8000, 16000]
    num_iterations = 24000
    num_hiddens = [dim, dim+10, dim+10, dim]


def get_config(name):
    try:
        return globals()[name+'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
