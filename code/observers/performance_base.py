import core
import numpy as np
import torch


class PerformanceObserverBase(core.Observer):

    def __init__(self, frequency=1):
        super().__init__(frequency)
        self.l2_squared_error = []
        self.l_infinity_true_support = []
        self.l_infinity_complement_support = []
        self.l1_norm = []
        self.validation_losses = []
        self.training_losses = []
        # Validation data set to None.
        # Will sample a new validation dataset once we have a handle to the
        # trainer.
        self.X_validation = None
        self.y_validation = None

    def append_performance_metrics(self, w_hat, trainer):
        self.append_oracle_metrics(w_hat, trainer)
        self.append_validation_metrics(w_hat, trainer)
        self.append_training_metrics(w_hat, trainer)

    def append_oracle_metrics(self, w_hat, trainer):
        # w_hat and w_star are both numpy arrays of same size.
        w_star = trainer.simulation_parameters.beta.flatten()
        k = trainer.simulation_parameters.k
        diff = w_hat.flatten() - w_star.flatten()

        self.l2_squared_error.append(np.sum(diff**2))
        self.l_infinity_true_support.append(np.max(np.absolute(diff[:k])))
        self.l_infinity_complement_support.append(
            np.max(np.absolute(diff[k:])))
        self.l1_norm.append(np.sum(np.absolute(w_hat)))

    def append_training_metrics(self, w_hat, trainer):
        training_loss = self.compute_loss(
            w_hat, trainer, trainer.X, trainer.y)
        self.training_losses.append(training_loss)

    def append_validation_metrics(self, w_hat, trainer):
        if self.X_validation is None:
            self.__sample_validation_data(trainer)
        validation_loss = self.compute_loss(
            w_hat, trainer, self.X_validation, self.y_validation)
        self.validation_losses.append(validation_loss)

    def compute_loss(self, w_hat, trainer, X, y):
        # Transform w_hat to torch tensor, so that the computation can be
        # performed on GPU if available.
        w_hat = w_hat.reshape(w_hat.size, 1)
        w_hat_tensor = torch.tensor(
            w_hat,
            device=trainer.pytorch_config.device,
            dtype=trainer.pytorch_config.dtype)

        y_pred = torch.matmul(X, w_hat_tensor)
        loss_criterion = trainer.model.get_loss_criterion()
        loss = loss_criterion(y_pred, y)
        return loss.item()

    def __sample_validation_data(self, trainer):
        n = trainer.simulation_parameters.dataset_size // 4
        if (trainer.X_validation is None):
            trainer.X_validation, trainer.y_validation = \
                trainer.dataset_generator.generate_data(
                    n, trainer.pytorch_config)
        self.X_validation = trainer.X_validation
        self.y_validation = trainer.y_validation

    def clean_up(self, trainer):
        self.X_validation = None
        self.y_validation = None
        trainer.X_validation = None
        trainer.y_validation = None

    @staticmethod
    def aggregate_performance_results(true_performance_observers):
        l2_squared_errors = core._aggregate_numeric_results(
            true_performance_observers, 'l2_squared_error')
        l_infty_S = core._aggregate_numeric_results(
            true_performance_observers, 'l_infinity_true_support')
        l_infty_Sc = core._aggregate_numeric_results(
            true_performance_observers, 'l_infinity_complement_support')
        l1_norm = core._aggregate_numeric_results(
            true_performance_observers, 'l1_norm')
        validation_losses = core._aggregate_numeric_results(
            true_performance_observers, 'validation_losses')
        training_losses = core._aggregate_numeric_results(
            true_performance_observers, 'training_losses')
        return {
            'l2_squared_errors': l2_squared_errors,
            'l_infty_S': l_infty_S,
            'l_infty_Sc': l_infty_Sc,
            'l1_norm': l1_norm,
            'validation_losses': validation_losses,
            'training_losses': training_losses
        }
