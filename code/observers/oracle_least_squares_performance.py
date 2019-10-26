import core
import numpy as np
import torch

from observers.performance_base import *


class OracleLeastSquaresPerformance(PerformanceObserverBase):
    """ A class for running LASSO by GLMNET for computing risks
    along a regularization path. """

    def __init__(self, frequency = 1):
        super().__init__(frequency)
        self.computed = False
        self.params = []

    def _notify(self, trainer):
        if not self.computed:
            k = trainer.simulation_parameters.k
            d = trainer.simulation_parameters.n_features

            # Subset the data matrix to the first k (true) parameters.
            X = trainer.X.cpu().detach().numpy()[:,:k]
            y = trainer.y.cpu().detach().numpy().squeeze()
            self.params.append(np.linalg.lstsq(X, y, rcond = -1)[0])
            padded_params = np.zeros(d)
            padded_params[:k] = self.params[0]
            self.params[0] = padded_params
            self.append_performance_metrics(self.params[0], trainer)

            self.computed = True

    @staticmethod
    def aggregate_results(oracle_ls_observers):
        results_dict =  PerformanceObserverBase \
            .aggregate_performance_results(oracle_ls_observers)
        params = core._aggregate_numeric_results(
                oracle_ls_observers, 'params')
        results_dict['params'] = params
        return results_dict

