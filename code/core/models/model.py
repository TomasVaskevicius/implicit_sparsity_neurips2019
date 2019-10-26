import torch
from abc import ABC, abstractmethod

class Model(ABC, torch.nn.Module):
    """ An abstract base class for model classes.
    This class specified the interface that has to be satisfied by all model
    classes and enforces them to also be of type torch.nn.module. """

    def __init__(self):
        ABC.__init__(self)
        torch.nn.Module.__init__(self)

    @abstractmethod
    def get_loss_criterion(self):
        """ A loss criterion to be used by the optimiser. """
        pass
