import time
import copy
import core
import utils as utils
from implicit_lasso_factory import \
    get_implicit_lasso_simulation
import numpy as np  # Output directory name.
import itertools

output_dir = './outputs/gd_vs_lasso_cor/'

# Devices (which GPUs) to use for running simulations.
device_ids = [0, 2, 3]
pytorch_configs = utils.get_pytorch_configs_from_device_ids(device_ids)

# How many simulations to perform on each device and using how many
# processes?
runs_per_device = 10
processes_per_device = 2

# The parameters that will be different among different simulations are
# alpha and noise_std.
d = 10000
k = 25

default_params = {
    'n_features': d,
    'k': k,
    'noise_std': 1.0,
    'observe_uv': 0,  # Do not save our parameterization.
    'observers_frequency': 10,
    'run_glmnet': 1,
    'store_glmnet_path': 0,
    'observe_parameters': 0,
    'use_alpha_oracle': 1,  # Do not tune alpha, use the one we provide.
    'alpha': 1e-12,
    'use_step_size_doubling_scheme': 1,
    'exponentiation_rate': 64,
    'epochs': 2000,
    'covariates': core.GaussianCovariates,
    'compute_oracle_ls': 1
}


def get_dataset_size_simulation_set_up():
    ns = np.exp(np.linspace(
        start=np.log(200), stop=np.log(5000), num=30))
    ns = ns.astype(int)
    gammas = np.array([0.25])
    sigmas = np.array([1.0])
    return ns, gammas, sigmas


def run(name, params_dict, mu=0.0):
    simulation_parameters = get_implicit_lasso_simulation(**params_dict)

    # Add mu attribute to simulation_parameters.
    setattr(simulation_parameters, 'mu', mu)
    # Add a correlation matrix if mu is not 0.0.
    if mu != 0.0:
        simulation_parameters.covariates_kwargs_dict['covariance_matrix'] = \
            mu * np.ones((d, d)) + (1.0 - mu) * np.identity(d)

    start = time.time()

    results = core.run_in_parallel(
        runs_per_device,
        processes_per_device,
        simulation_parameters,
        pytorch_configs)

    # Do not save the covariance matrix.
    simulation_parameters.covariates_kwargs_dict = {}

    utils.save_simulation_output(
        simulation_name, results, simulation_parameters)
    end = time.time()
    print("Execution time for id: ", run_id, ":",
          end - start, "seconds.")


if __name__ == '__main__':
    run_id = 0

    simulation_set_ups = [
        get_dataset_size_simulation_set_up
    ]

    for simulation_set_up in simulation_set_ups:
        ns, gammas, sigmas = simulation_set_up()
        for n, gamma, sigma in itertools.product(ns, gammas, sigmas):
            w_star = np.zeros((d, 1))
            w_star[:k, 0] = np.ones(k) * gamma
            params = copy.deepcopy(default_params)
            params['dataset_size'] = n
            params['batch_size'] = n
            params['beta'] = w_star
            params['noise_std'] = sigma
            # Run simulation for mu = 0.5.
            simulation_name = output_dir + "run_" + str(run_id)
            run(simulation_name, params, mu=0.5)
            run_id += 1
            # Run simulation for mu = 0.0.
            simulation_name = output_dir + "run_" + str(run_id)
            run(simulation_name, params, mu=0.0)
            run_id += 1
