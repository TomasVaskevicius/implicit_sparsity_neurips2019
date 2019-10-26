import time
import copy
import core
import utils as utils
from implicit_lasso_factory import \
    get_implicit_lasso_simulation
import numpy as np  # Output directory name.
import itertools

output_dir = './outputs/gd_vs_lasso/'

# Devices (which GPUs) to use for running simulations.
device_ids = [5, 7]
pytorch_configs = utils.get_pytorch_configs_from_device_ids(device_ids)

# How many simulations to perform on each device and using how many
# processes?
runs_per_device = 15
processes_per_device = 5

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
    'covariates': core.RademacherCovariates,
    'compute_oracle_ls': 1
}


def get_signal_size_simulation_set_up():
    ns = np.array([500])
    gammas = np.linspace(start=0.0, stop=1.0, num=30)
    sigmas = np.array([1.0])
    return ns, gammas, sigmas


def get_sigma_simulation_set_up():
    ns = np.array([500])
    gammas = np.array([1.0])
    sigmas = np.linspace(start=0.0, stop=12.0, num=30)
    return ns, gammas, sigmas


def get_dataset_size_simulation_set_up():
    ns = np.exp(np.linspace(
        start=np.log(200), stop=np.log(5000), num=30))
    ns = ns.astype(int)
    gammas = np.array([0.25])
    sigmas = np.array([1.0])
    return ns, gammas, sigmas


def run(name, params_dict, paths_simulation=False):
    simulation_parameters = get_implicit_lasso_simulation(**params_dict)
    start = time.time()

    if paths_simulation:
        # This is the last simulation, the one used for plotting gd and
        # lasso paths.
        results = core.run_in_parallel(
            1,
            1,
            simulation_parameters,
            [pytorch_configs[0]])
    else:
        results = core.run_in_parallel(
            runs_per_device,
            processes_per_device,
            simulation_parameters,
            pytorch_configs)

    utils.save_simulation_output(
        simulation_name, results, simulation_parameters)
    end = time.time()
    print("Execution time for id: ", run_id, ":",
          end - start, "seconds.")


if __name__ == '__main__':
    run_id = 0

    simulation_set_ups = [
        get_signal_size_simulation_set_up,
        get_sigma_simulation_set_up,
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
            simulation_name = output_dir + "run_" + str(run_id)
            run(simulation_name, params)
            run_id += 1

    # Perform a final simulation for comparing gd and lasso paths.
    params = copy.deepcopy(default_params)
    w_star = np.zeros((d, 1))
    w_star[:k, 0] = np.ones(k) * 1.0  # gamma = 1.0
    params['dataset_size'] = 500
    params['batch_size'] = 500
    params['beta'] = w_star
    params['noise_std'] = 1.0
    params['observe_parameters'] = 1
    params['store_glmnet_path'] = 1
    simulation_name = output_dir + "run_" + str(run_id)
    run(simulation_name, params, paths_simulation=True)
