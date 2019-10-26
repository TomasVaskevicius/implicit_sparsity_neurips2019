import core
import pickle
import torch


def save_simulation_output(name, results, simulation_parameters_cls, **kwargs):
    """ A helper function to save simulation outputs together with arbitrary
    metadata. """
    output = {
        'name': name,
        'results': results,
        'simulation_parameters_cls': simulation_parameters_cls}
    output = dict(output, **kwargs)
    with open(name + '.pickle', 'wb') as _file:
        pickle.dump(output, _file)


def load_simulation_output(name):
    """ Loads the simulation output saved by save_simulation_output. """
    with open(name + '.pickle', 'rb') as _file:
        output = pickle.load(_file)
    return output


def get_pytorch_configs_from_device_ids(device_ids_list):
    return [core.PyTorchConfig(torch.float32, 'cuda:' + str(device_id))
            for device_id in device_ids_list]
