from abc import ABC, abstractmethod

from ..config.config import PyTorchConfig
from ..observers.observable import Observable

class Trainer(Observable):
    """ An abstract base class for all runners.
    It enforces all trainers to support observers and require constructors
    to take model and dataset_generator as arguments. """

    def __init__(
            self,
            n,
            dataset_generator,
            model,
            pytorch_config = PyTorchConfig()):
        """
        n is the dataset size.
        dataset_generator should be a subclass of core.data.dataset.
        model should be a subclass of core.model.model.
        """
        super().__init__()
        X, y = dataset_generator.generate_data(n, pytorch_config)
        self.X = X
        self.y = y
        self.X_validation = None
        self.y_validation = None
        self.dataset_generator = dataset_generator
        self.model = model
        self.pytorch_config = pytorch_config
        dtype = pytorch_config.dtype
        device = pytorch_config.device
        self.model.to(dtype = dtype, device = device)

    def finish(self):
        """ Should be called after the training is done to remove the dataset
        from memory. """
        # Clean up all observers.
        self.clean_up_all_observers()
        # To avoid memory issues, any of the above if needed should be saved
        # and processed by observers.
        self.X = None
        self.y = None
        self.model = None

    @abstractmethod
    def do_one_epoch(self):
        """ Performs one epoch of training some model. """
        pass
