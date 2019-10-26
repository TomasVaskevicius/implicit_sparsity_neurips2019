import time
import copy
import core
import utils as utils
from implicit_lasso_factory import \
    get_implicit_lasso_simulation
import numpy as np

# Output directory name.
output_dir = './outputs/exponential_convergence/'

# Devices (which GPUs) to use for running simulations.
device_ids = [5, 7]
pytorch_configs = utils.get_pytorch_configs_from_device_ids(device_ids)

# How many simulations to perform on each device and using how many
# processes?
runs_per_device = 15
processes_per_device = 2

# The parameters that will be different among different simulations are
# alpha and noise_std.
n = 250
d = 10000
k = 7
np.random.seed(0)
w_star = np.zeros((d, 1))
w_star[:k, 0] = 2.0**(np.arange(k, dtype=float))
default_params = {
    'n_features': d,
    'dataset_size': n,
    'batch_size': n,  # Use gradient descent instead of SGD.
    'k': k,
    'beta': w_star,
    'noise_std': 1.0,
    'observe_uv': 0,  # Do not save our parameterization.
    'observers_frequency': 10,
    'run_glmnet': 0,
    'observe_parameters': 0,
    'use_alpha_oracle': 1,  # Do not tune alpha, use the one we provide.
    'alpha': 1e-12,
    'use_eta_oracle': 1,  # Do not tune the step size.
    'learning_rate': 1.0 / (20.0 * np.max(w_star)),
    'use_wmax_oracle': 1,
    'w_max_oracle': np.max(w_star),
    'compute_oracle_ls': 0,
    'covariates': core.RademacherCovariates
}


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
    print("Starting simulations with increasing step sizes.")
    run_id = 0
    params = copy.deepcopy(default_params)
    params['use_step_size_doubling_scheme'] = 1
    params['exponentiation_rate'] = 64
    params['epochs'] = 2500
    simulation_name = output_dir + "run_" + str(run_id)
    run(simulation_name, params)

    print("Starting simulations with constant step sizes.")
    run_id += 1
    params = copy.deepcopy(default_params)
    params['use_step_size_doubling_scheme'] = 0
    params['epochs'] = 40000
    simulation_name = output_dir + "run_" + str(run_id)
    run(simulation_name, params)

    # Save paths for one simulation with both settings
    runs_per_device = 1

    print("Starting simulations with increasing step sizes for storing paths.")
    run_id += 1
    params = copy.deepcopy(default_params)
    params['use_step_size_doubling_scheme'] = 1
    params['exponentiation_rate'] = 64
    params['epochs'] = 2500
    params['observe_parameters'] = 1
    simulation_name = output_dir + "run_" + str(run_id)
    run(simulation_name, params)

    print("Starting simulations with constant step sizes.")
    run_id += 1
    params = copy.deepcopy(default_params)
    params['use_step_size_doubling_scheme'] = 0
    params['epochs'] = 40000
    params['observe_parameters'] = 1
    simulation_name = output_dir + "run_" + str(run_id)
    run(simulation_name, params)
