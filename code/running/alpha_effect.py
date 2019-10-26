import time
import core
import utils as utils
from implicit_lasso_factory import \
    get_implicit_lasso_simulation
import numpy as np

# Output directory name.
output_dir = './outputs/alpha_effect/'

# Devices (which GPUs) to use for running simulations.
device_ids = [3, 6]
pytorch_configs = utils.get_pytorch_configs_from_device_ids(device_ids)

# How many simulations to perform on each device and using how many
# processes?
runs_per_device = 15
processes_per_device = 2

# The parameters that will be different among different simulations are
# alpha and noise_std.
n = 100
d = 10000
k = 5
np.random.seed(0)
w_star = np.zeros((d, 1))
w_star[:k, 0] = np.ones(k)
default_params = {
    'n_features': d,
    'dataset_size': n,
    'batch_size': n,  # Use gradient descent instead of SGD.
    'k': k,
    'beta': w_star,
    'epochs': 1000,
    'learning_rate': 0.05,
    'observe_uv': 0,  # Do not save our parameterization.
    'use_alpha_oracle': 1,  # Do not tune alpha, use the one we provide.
    'use_eta_oracle': 1,  # Do not tune the step size, use the provided one.
    'use_step_size_doubling_scheme': 0,  # Use constant step size.
    'compute_oracle_ls': 0,
    'covariates': core.RademacherCovariates
}

if __name__ == '__main__':
    alphas = 10.0**(np.array([-2.0, -3.0, -4.0]))
    noise_stds = [0.5]
    run_id = 0
    print("Starting simulations for different alphas")
    for alpha in alphas:
        for noise_std in noise_stds:
            simulation_name = output_dir + "alphas/run_" + str(run_id)
            simulation_parameters = get_implicit_lasso_simulation(
                alpha=alpha, observe_parameters=0,
                noise_std=noise_std, observers_frequency=10,
                run_glmnet=0, store_glmnet_path=0,
                **default_params)
            # Start our simulations.
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
            run_id += 1

    # Now we perform simulatinos for saving the parameter paths.
    alphas = 10.0**(np.array([-3.0, -12.0]))
    run_id = 0
    print("Starting simulations for saving parameter paths")
    for alpha in alphas:
        for noise_std in noise_stds:
            simulation_name = output_dir + "paths/run_" + str(run_id)
            simulation_parameters = get_implicit_lasso_simulation(
                alpha=alpha, noise_std=noise_std,
                observe_parameters=1, observers_frequency=1,
                run_glmnet=1, store_glmnet_path=1,
                **default_params)
            # Start our simulations.
            start = time.time()
            results = core.run_in_parallel(
                1,  # Only one run for each choice of alpha is enough here.
                1,  # Only one process per device.
                simulation_parameters,
                [pytorch_configs[0]])  # Use the first device.
            utils.save_simulation_output(
                simulation_name, results, simulation_parameters)
            end = time.time()
            print("Execution time for id: ", run_id, ":",
                  end - start, "seconds.")
            run_id += 1
