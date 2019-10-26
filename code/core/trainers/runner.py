import numpy as np
import random
import torch

from .sgd_trainer import SgdTrainer, SgdHyperparameters
from ..config.config import PyTorchConfig
from ..observers.observable import Observable


def _run_sgd(simulation_parameters, pytorch_config):
    """ An actual implementation of the run_sgd method. Here we do not
        need to worry about setting the seed or switching to the right
        cuda device if necessary.
    """

    sgd_hyperparameters = SgdHyperparameters(
            simulation_parameters.learning_rate,
            simulation_parameters.batch_size)
    sgd_trainer = SgdTrainer(
            simulation_parameters.dataset_size,
            simulation_parameters.get_dataset_generator(),
            simulation_parameters.get_model(),
            sgd_hyperparameters,
            pytorch_config)

    sgd_trainer.simulation_parameters = simulation_parameters

    observers_list = simulation_parameters.get_observers_list()
    for observer in observers_list:
        sgd_trainer.register_observer(observer)

    for i in range(simulation_parameters.epochs):
        sgd_trainer.do_one_epoch()

    sgd_trainer.finish()

    return sgd_trainer.observers

def run_sgd(simulation_parameters, pytorch_config, seed):
    """ A function for running SGD for the simulation specified by the
        simulation parameters obect. Seed sets numpy and pytorch seeds
        to the specified value for reproducibility purposes.
    """

    # See https://github.com/pytorch/pytorch/issues/7068.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    if pytorch_config.device == 'cpu':
        results = _run_sgd(
                simulation_parameters,
                pytorch_config)
    else:
        device_id = int(pytorch_config.device.split(':')[1])
        with torch.cuda.device(device_id):
            results = _run_sgd(
                    simulation_parameters,
                    pytorch_config)

    return results

