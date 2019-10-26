import time
import copy
import core
import utils as utils
from implicit_lasso_factory import \
    get_implicit_lasso_simulation
import numpy as np  # Output directory name.

output_dir = './outputs/gd_vs_lasso_d/'

# Devices (which GPUs) to use for running simulations.
device_ids = [5, 7]
pytorch_configs = utils.get_pytorch_configs_from_device_ids(device_ids)

# How many simulations to perform on each device and using how many
# processes?
runs_per_device = 15
processes_per_device = 10

# The parameters that will be different among different simulations are
# alpha and noise_std.
k = 25
n = 500

default_params = {
    'dataset_size': n,
    'batch_size': n,
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


def get_d_simulation_set_up():
    ds = np.exp(np.linspace(
        start=np.log(500), stop=np.log(10000), num=30))
    ds = ds.astype(int)
    return ds


def run(name, params_dict):
    simulation_parameters = get_implicit_lasso_simulation(**params_dict)
    start = time.time()
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

    for d in get_d_simulation_set_up():
        params = copy.deepcopy(default_params)
        params['n_features'] = d
        w_star = np.zeros((d, 1))
        w_star[:k, 0] = np.ones(k) * 1.0
        params['beta'] = w_star
        simulation_name = output_dir + "run_" + str(run_id)
        run(simulation_name, params)
        run_id += 1
