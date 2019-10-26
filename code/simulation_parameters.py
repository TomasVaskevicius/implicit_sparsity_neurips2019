import core
from model import ImplicitLassoLinearRegression
from observers.gd_performance import *
from observers.lasso_performance import *
from observers.hyperparameters_tuning import *
from observers.parameters_observers import *
from observers.oracle_least_squares_performance import *


class ImplicitLassoSimulationParameters(core.SimulationParameters):
    """ A class for defining our simulation parameters, together with what
    dataset and model should be used. """

    def __init__(self):
        super().__init__()
        self.beta = None  # Ground truth parameter.
        self.noise_std = None
        self.k = None  # Sparsity parameter.
        self.alpha = None  # Initialization size.
        self.observers_frequency = 1  # Can be overriden by command line args.
        # In the following parameter 0 is False, while 1 is True.
        self.run_glmnet = 0
        self.store_glmnet_path = 0
        self.observe_parameters = 0
        self.observe_uv = 0
        # Parameters for automatic parameter tuning observer
        self.epsilon = 1e-3
        self.use_alpha_oracle = 0
        self.use_eta_oracle = 0
        self.use_wmax_oracle = 0
        self.use_step_size_doubling_scheme = 0
        # These params are reset during parameter tuning observer
        # initialization.
        self.eta_oracle = None
        self.alpha_oracle = None
        self.w_max_oracle = None
        # The following parameter shows how much faster we want to do
        # exponentiation than the factor 640 suggested in our proofs.
        self.exponentiation_rate = 1
        self.store_masks = True
        self.covariates = core.GaussianCovariates
        self.covariates_kwargs_dict = {}
        # Whether to compute the oracle least squares solutions.
        self.compute_oracle_ls = 0
        # Default model to use.
        self.model_class = ImplicitLassoLinearRegression

    def get_dataset_generator(self):
        """ beta is the true model parameters for simulating the data. """
        dataset_generator = core.LinearRegressionDataset(
            self.n_features, self.noise_std, self.covariates_kwargs_dict)
        dataset_generator.covariates = self.covariates
        dataset_generator.beta = self.beta
        return dataset_generator

    def get_model(self):
        model = self.model_class(self.n_features)
        model.init_weights(self.alpha)
        return model

    def get_observers_list(self):
        frequency = self.observers_frequency
        observers_list = []

        # Add performance metrics observer.
        observers_list.append(TruePerformanceObserver(frequency))

        # Other observers are added conditionally on the command line
        # arguments.
        if self.observe_parameters is 1:
            observers_list.append(ParametersObserver(frequency))
        if self.observe_uv is 1:
            observers_list.append(UVObserver(frequency))
        if self.run_glmnet is 1:
            if self.store_glmnet_path is 1:
                store_path = True
            else:
                store_path = False
            observers_list.append(LassoPathObserver(frequency, store_path))
        if self.compute_oracle_ls is 1:
            observers_list.append(OracleLeastSquaresPerformance(frequency))

        # Set up parameters for automatic hyperparmeters observer.
        if self.use_wmax_oracle is 1:
            self.w_max_oracle = np.max(np.absolute(self.beta))
        if self.use_alpha_oracle is 1:
            self.alpha_oracle = self.alpha
        if self.use_eta_oracle is 1:
            self.eta_oracle = self.learning_rate
        if self.use_step_size_doubling_scheme is 1:
            self.double_step_sizes = True
        else:
            self.double_step_sizes = False
        observers_list.append(ImplicitLassoAutomaticHyperparameters(
            self.epsilon,
            self.double_step_sizes,
            self.w_max_oracle,
            self.alpha_oracle,
            self.eta_oracle))

        return observers_list
