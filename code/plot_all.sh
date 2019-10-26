#!/bin/bash

set -x

python -m plotting.alpha_effect_output_handler
python -m plotting.exponential_convergence_output_handler
python -m plotting.gd_vs_lasso_output_handler
python -m plotting.gd_vs_lasso_d_output_handler
python -m plotting.gd_vs_lasso_cor_output_handler
python -m plotting.n_vs_k_output_handler
