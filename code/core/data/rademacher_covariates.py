import numpy as np
import torch

from ..config.config import PyTorchConfig

class RademacherCovariates:
    """ A class for generating normally distributed covariates. """

    @staticmethod
    def generate_covariates_matrix(
            n, d, pytorch_config = PyTorchConfig()):
        """ Parameters:
            n The number of data points.
            d The number of covariates per data point.
            pytorch_config A gen.config.PyTorchConfig object. """
        dtype = pytorch_config.dtype
        device = pytorch_config.device

        mean = np.zeros(d)
        samples = np.random.binomial(1, 0.5, (n, d)) * 2 - 1
        samples = torch.tensor(samples, dtype = dtype, device = device)
        return samples

