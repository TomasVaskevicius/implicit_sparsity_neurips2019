import core
import numpy as np
import torch


class ImplicitLassoAutomaticHyperparameters(core.Observer):
    """ A class for tuning hyperparameters. """

    def __init__(
            self,
            epsilon=1e-3,
            increase_step_sizes=True,
            w_max_oracle=None,
            alpha_oracle=None,
            eta_oracle=None,
            store_masks=True):
        """
        epsilon The desired maximum squared l2 norm parameter estimation
            error. Could not be reache if the noise is too high,
            but if the noise is low enough then it is guaranteed to be reached.
        increase_step_sizes Tells whether the increasing step sizes scheme
            should_be_used.
        w_max_oracle A lower-bound on w_max. Should be only used if one can
            estimate w_max with multiplicative constant better than 2.
            Otherwise should be left as None.
        alpha_oracle Will override estimation of alpha if set.
        eta_oracle Will override setting of step size if set.
        store_masks A flag indicating whether to store step size multiplication
            masks during each inductino phase.
        """
        super().__init__(1)
        self.epsilon = epsilon
        self.increase_step_sizes = increase_step_sizes
        self.w_max_oracle = w_max_oracle
        self.alpha_oracle = alpha_oracle
        self.eta_oracle = eta_oracle
        self.iter_id = -2
        self.completed_induction_steps = 0
        self.store_masks = store_masks
        self.masks = []

    def clean_up(self, trainer):
        # Keeping pytorch cuda tensors causes problems with multiprocessing.
        self.mask = None

    def _notify(self, trainer):
        if self.iter_id == -2:
            self.__handle_first_iteration(trainer)
        elif self.iter_id == -1:
            self.__handle_second_iteration(trainer)
            self.__set_up_backward_hook(trainer)
        elif self.iter_id == self.induction_phase_t:
            self.iter_id = -1
            self.completed_induction_steps += 1
            if self.completed_induction_steps >= 2:
                self.__update_step_size_mask(trainer)
        self.iter_id += 1

    def __handle_first_iteration(self, trainer):
        # Assume that w_max >= 1e-10
        self.lr = 0.2 * 1e-3
        trainer.lr = self.lr
        # Initialize the weights for the first iteration.
        trainer.model.init_weights(1)

    def __handle_second_iteration(self, trainer):
        # Estimate w_max.
        u_numpy = trainer.model.layer.weight.cpu().detach().numpy()
        v_numpy = trainer.model.layer.weight.cpu().detach().numpy()
        f_max = np.max(np.concatenate((u_numpy, v_numpy)))
        self.w_max_hat = (f_max - 1.0) / (3.0 * self.lr)

        # We have w_max <= w_max_hat < 2 * w_max.
        # We can now reset the initialization and learning rate in the training
        # algorithm.

        # Estimate new alpha and set it to the model.
        trainer.lr = 1.0 / (20.0 * self.w_max_hat)
        self.d = u_numpy.size
        self.alpha = np.min([
            (2.0 * self.epsilon) / (2.0 * self.d + 1),
            self.epsilon / self.w_max_hat**2,
            1.0 / (2.0 * self.d)])  # (w^{*}_{min} \wedge 1) / 2 condition.

        # Override alpha if oracle is given.
        if self.alpha_oracle is not None:
            self.alpha = self.alpha_oracle
        # Override the step size if oracle is given.
        if self.eta_oracle is not None:
            trainer.lr = self.eta_oracle
        # Override w_max_hat if oracle is provided.
        if self.w_max_oracle is not None:
            self.w_max_hat = self.w_max_oracle

        # Compute how long one induction phase takes.
        # Note that we adjust the constant 640 by dividing it with
        # exponentiation rate as specified in simulation parameters.
        self.induction_phase_t = \
            (640 // trainer.simulation_parameters.exponentiation_rate) \
            * np.ceil(np.log(1.0 / self.alpha)) \

        # Set new alpha.
        trainer.model.init_weights(self.alpha)

        # For logging purposes, store what parameters were used.
        # We wrap it in a list so that our results agreggator can be used.
        self.used_eta = [trainer.lr]
        self.used_alpha = [self.alpha]
        self.used_w_max_hat = [self.w_max_hat]

    def __set_up_backward_hook(self, trainer):
        # Set up a backward hook for multiplying gradients with self.mask.
        self.mask = torch.ones_like(trainer.model.layer.weight.data)

        def hook(grad):
            grad *= self.mask
        trainer.model.layer.weight.register_hook(hook)
        trainer.model.layer2.weight.register_hook(hook)

    def __update_step_size_mask(self, trainer):
        u_numpy = trainer.model.layer.weight.cpu().detach().numpy().flatten()
        v_numpy = trainer.model.layer.weight.cpu().detach().numpy().flatten()

        u2 = u_numpy**2
        v2 = v_numpy**2

        cutoff = 2**(-1-self.completed_induction_steps) * self.w_max_hat
        update_idx = np.logical_and(u2 <= cutoff, v2 <= cutoff)
        update_idx = np.argwhere(update_idx).flatten()
        if self.increase_step_sizes:
            self.mask[0, update_idx] *= 2.0

        if (self.store_masks):
            self.masks.append(self.mask.cpu().detach().numpy().flatten())

    @staticmethod
    def aggregate_results(parameter_tuning_observers):
        etas = core._aggregate_numeric_results(
            parameter_tuning_observers, 'used_eta')
        alphas = core._aggregate_numeric_results(
            parameter_tuning_observers, 'used_alpha')
        w_max_hats = core._aggregate_numeric_results(
            parameter_tuning_observers, 'used_w_max_hat')
        masks = core._aggregate_numeric_results(
            parameter_tuning_observers, 'masks')

        output_dict = {
            'etas': etas.squeeze(),
            'alphas': alphas.squeeze(),
            'w_max_hats': w_max_hats.squeeze(),
            'masks': masks.squeeze()
        }

        return output_dict
