import copy
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import itertools

from .runner import run_sgd
from .sgd_trainer import SgdTrainer, SgdHyperparameters
from ..config.config import PyTorchConfig
from ..observers.observable import Observable


def job(
        job_id,
        simulation_parameters,
        pytorch_config):
    """ A job running the specified simulation on a given device. """

    simulation = copy.deepcopy(simulation_parameters)
    simulation.pytorch_config = pytorch_config
    simulation.seed = job_id
    simulation.run()
    return simulation.output


def _process_all_outputs(observers_list_list, results_dict = {}):
    """ A helper function for aggregating results of all observers. The input
    to this function should be the output of run_in_parallel function. """

    if observers_list_list == []:
        return results_dict

    # A list of list of observers aggregated by type.
    aggregated_observers_list = []

    # This will be the list of observers list that will not be processed in
    # this function calls - that is, observers of the observers at the current
    # top level.
    unprocessed_observers_list_list = []

    # Set up an empty list for each observer at the top level.
    for observer in observers_list_list[0]:
        aggregated_observers_list.append([])

    for observers_list in observers_list_list:
        unprocessed_observers_list = []

        for i, observer in enumerate(observers_list):
            aggregated_observers_list[i].append(observer)
            # If the current observer is also observable, add its observers
            # to the unprocessed_observers_list.
            if issubclass(type(observer), Observable):
                unprocessed_observers_list += observer.observers

        if unprocessed_observers_list != []:
            unprocessed_observers_list_list.append(unprocessed_observers_list)

    for aggregated_observers in aggregated_observers_list:
        name = type(aggregated_observers[0]).__name__
        result = type(aggregated_observers[0]).aggregate_results(
                aggregated_observers)
        if result is not None:
            results_dict[name] = result

    return _process_all_outputs(unprocessed_observers_list_list, results_dict)


def run_in_parallel(
        times_per_device,
        n_processes_per_device,
        simulation_parameters,
        pytorch_configs,
        process_initialiser = None,
        initialiser_args = ()):
    """ Executes the set up experiment on the given devices.
        Parameters:

        times The number of times the given experiment has to be repeated on
                each device.
        n_processes_per_device The number of independent processes to be used
                for each pytorch_config.
        simulation_parameters An instance of subclass of SimulationParameters
                class.
        observers_factory A factory method for the required observers.
        pytorch_configs A list of PyTorchConfig objects specifying the devices
                and data types to be used. For example, this list could contain
                two gpus, then, setting n_processes = 2 (or more) both gpus
                will be used to execute the simulations.
        process_initialiser Each process will be initialised by calling this
                function.
        process_initialiser_args Arguments to be passed to the initialiser
                function. This parameter must be a list, providing the
                arguments for each different device. """

    multiprocessing.set_sharing_strategy('file_system')
    context = multiprocessing.get_context('spawn')

    pools = []

    for config, initargs in \
            itertools.zip_longest(pytorch_configs, initialiser_args):
        pools.append(context.Pool(
                processes = n_processes_per_device,
                initializer = process_initialiser,
                initargs = initargs))

    results = []
    for pool_id, (pool, config) in enumerate(zip(pools, pytorch_configs)):
        args = []
        for i in range(times_per_device):
            args.append((
                    pool_id * times_per_device + i,
                    simulation_parameters,
                    config))
        results.append(pool.starmap_async(job, args))

    all_outputs = []
    for pool, result in zip(pools, results):
        all_outputs += result.get()
        pool.close()
        pool.join()

    return _process_all_outputs(all_outputs)

