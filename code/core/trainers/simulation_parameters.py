from abc import ABC, abstractmethod
import copy

from .runner import run_sgd

class SimulationParameters(ABC):
    """ Should be extended by other classes defining particular experiments. """

    def __init__(self):
        self.n_features = 10
        self.dataset_size = 10
        self.learning_rate = 0.001
        self.batch_size = 1
        self.epochs = 1
        self.seed = 1
        self.pytorch_config = None

    def run(self):
        self.output = run_sgd(self, self.pytorch_config, self.seed)

    def __get_modified_dict(self, simulation_parameters):
        # Two sp classses are equivalent if all but its
        # 'output', 'seed' and 'pytorch_config' attributes are the same.
        sp_dict = copy.deepcopy(simulation_parameters.__dict__)
        sp_dict.pop('output', None)
        sp_dict.pop('seed', None)
        sp_dict.pop('pytorch_config', None)
        return sp_dict

    def __eq__(self, other):
        self_dict = self.__get_modified_dict(self)
        other_dict = self.__get_modified_dict(other)
        return str(self_dict) == str(other_dict)

    def __hash__(self):
        self_dict = self.__get_modified_dict(self)
        return hash(frozenset(self_dict))

    @abstractmethod
    def get_dataset_generator(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_observers_list(self):
        pass


