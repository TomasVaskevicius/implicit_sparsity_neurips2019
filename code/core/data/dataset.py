from abc import ABC, abstractmethod

from ..config.config import PyTorchConfig

class Dataset(ABC):
    """ An abstract base class for dataset generators. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_data(self, n, pytorch_config = PyTorchConfig()):
        """ Returns a dataset in the form of X, y, where X is a matrix of n
        rows containing the covariates and y is the output.
        Both X and y should be torch tensords, where the datatype and device is
        specified by the pytorch_config parameter of type PyTorchConfig."""
        pass
