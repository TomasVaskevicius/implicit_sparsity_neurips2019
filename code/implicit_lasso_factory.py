import core
from simulation_parameters import \
    ImplicitLassoSimulationParameters


def get_implicit_lasso_simulation(
        dataset_size=1000,
        n_features=10000,
        learning_rate=0.01,
        batch_size=None,
        epochs=1000,
        seed=1,
        beta=None,
        noise_std=1.0,
        k=10,
        alpha=1e-14,
        observers_frequency=10,
        run_glmnet=0,
        store_glmnet_path=0,
        observe_parameters=0,
        observe_uv=0,
        epsilon=1e-6,
        use_step_size_doubling_scheme=0,
        use_alpha_oracle=0,
        use_eta_oracle=0,
        use_wmax_oracle=0,
        w_max_oracle=None,
        eta_oracle=None,
        alpha_oracle=None,
        exponentiation_rate=1,
        store_masks=True,
        pytorch_config=None,
        covariates=core.GaussianCovariates,
        compute_oracle_ls=0):
    params_dict = locals()
    simulation = ImplicitLassoSimulationParameters()
    simulation.__dict__.update(params_dict)
    return simulation
