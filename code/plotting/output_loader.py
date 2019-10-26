import os
from utils import load_simulation_output


def add_simulation_output(fname, outputs_dict):
    outputs = load_simulation_output(fname)
    results = outputs['results']

    processed_output = {
        'gd_performance': results['TruePerformanceObserver'],
        'tuned_params': results['ImplicitLassoAutomaticHyperparameters'],
        'simulation_parameters': outputs['simulation_parameters_cls']
    }

    # Check which of the optional observers, if any, were used.
    if 'LassoPathObserver' in results.keys():
        processed_output['lasso_performance'] = \
            results['LassoPathObserver']
    if 'ParametersObserver' in results.keys():
        processed_output['params'] = results['ParametersObserver']
    if 'UVObserver' in results.keys():
        processed_output['uv'] = results['UVObserver']
    if 'OracleLeastSquaresPerformance' in results.keys():
        processed_output['oracle_ls'] = \
            results['OracleLeastSquaresPerformance']

    outputs_dict[outputs['simulation_parameters_cls']] = processed_output


def load_simulations_from_directory(simulations_dir):
    simulations = {}
    for fname_dot_pickle in os.listdir(simulations_dir):
        fname = simulations_dir + str.split(fname_dot_pickle, '.')[0]
        if fname.endswith('readme'):
            continue
        add_simulation_output(fname, simulations)
    return simulations
