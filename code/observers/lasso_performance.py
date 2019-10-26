import core
import numpy as np
from glmnet import ElasticNet

from observers.performance_base import *


class LassoPathObserver(PerformanceObserverBase):
    """ A class for running LASSO by GLMNET for computing risks
    along a regularization path. """

    def __init__(self, frequency=1, store_path=False):
        super().__init__(frequency)
        self.store_path = store_path
        self.computed = False
        self.lambdas = []
        self.coefs = []

    def _notify(self, trainer):
        if not self.computed:
            X = trainer.X.cpu().detach().numpy()
            y = trainer.y.cpu().detach().numpy()

            glmnet = ElasticNet(
                n_splits=0,
                fit_intercept=False,
                lambda_path=np.exp(np.linspace(
                    start=np.log(10**(-9)),
                    stop=np.log(10),
                    num=200)))
            glmnet.fit(X, y.squeeze())
            self.lambdas, self.coefs = glmnet.lambda_path_, glmnet.coef_path_
            # Swap axes for parameters learned by lasso path, so that lambda
            # corresponds to the same axis as time for gradient descent iterates.
            self.coefs = np.swapaxes(self.coefs, 0, 1)
            # So now self.coefs is of shape (n_lambda, n_params).
            for lambda_id in range(self.coefs.shape[0]):
                w_lambda = self.coefs[lambda_id, :]
                self.append_performance_metrics(w_lambda, trainer)

            if self.store_path is False:
                # Dummy array so that _aggregate_numeric_results works.
                self.coefs = [np.array([[-1], [-1]])]

            self.computed = True

    @staticmethod
    def aggregate_results(lasso_path_observers):
        results_dict = PerformanceObserverBase \
            .aggregate_performance_results(lasso_path_observers)

        lambdas = core._aggregate_numeric_results(
            lasso_path_observers, 'lambdas')
        coef_paths = core._aggregate_numeric_results(
            lasso_path_observers, 'coefs')

        results_dict['lambdas'] = lambdas
        results_dict['coef_paths'] = coef_paths

        return results_dict
