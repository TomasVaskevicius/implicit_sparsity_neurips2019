import numpy as np
import torch

from ..config.config import PyTorchConfig

class GaussianCovariates:
    """ A class for generating normally distributed covariates. """

    @staticmethod
    def generate_covariates_matrix(
            n, d, pytorch_config = PyTorchConfig(), covariance_matrix = None):
        """ Parameters:
            n The number of data points.
            d The number of covariates per data point.
            covariance_matrix A d x d numpy covariance matrix. If none,
                    identity matrix is assumed.
            pytorch_config A gen.config.PyTorchConfig object. """
        dtype = pytorch_config.dtype
        device = pytorch_config.device

        if covariance_matrix is None:
            # Assume identity covariance matrix.
            return torch.randn(n, d, dtype = dtype, device = device)

        if list(covariance_matrix.shape) != [d, d]:
            raise ValueError("The covariance Matrix should be " \
                    + str(d) + " x " + str(d) + ".")

        mean = torch.zeros(d, dtype = dtype, device = device)
        covariance_matrix = torch.tensor(
                covariance_matrix, device = device, dtype = dtype)

        multivar_norm = torch.distributions.MultivariateNormal(
                covariance_matrix = covariance_matrix, loc = mean)

        samples = torch.zeros(n, d, device = device, dtype = dtype)
        # The below way to filling up the dataset is to save cuda memory.
        # torch multivariate normal rng somehow uses up too much memory.
        for i in range(n):
            samples[i,:] = multivar_norm.rsample((1,))

        return samples

