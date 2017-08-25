""" Default configuration and hyperparameter values for dynamics objects. """
from icra import __file__ as icra_filepath
# from icra.src.dynamics import DynamicsPriorGMM, DynamicsLRPrior
# from dynamics import * #DynamicsPriorGMM, DynamicsLRPrior

# DynamicsPriorGMM
DYN_PRIOR_GMM = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
    'initial_condition': 4,
    'T': 8,
    'regularization': 1e-6,
    'sample_size': 542,
}

DYNAMICS_PROPERTIES = {
    # 'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        # 'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
    #Traj settings
    'initial_state_var':1e-6,
}
