import matplotlib as mpl
import numpy as np
import copy
import itertools

from running.gd_vs_lasso_cor \
    import get_dataset_size_simulation_set_up
from plotting.output_loader \
    import load_simulations_from_directory
from plotting.output_utils import \
    find_optimal_gd_values, find_optimal_lasso_values
from plotting.plotting_params import \
    legend_font_size, font_size, get_new_fig_and_ax, \
    marker_size, line_width, save_fig


def plot_oracle_errors(simulations, ax, simulation_set_up, mu):
    sp = list(simulations.keys())[0]
    sp = copy.deepcopy(sp)
    sp.mu = mu

    ns, gammas, sigmas = simulation_set_up()

    if ns.size > 1:
        xaxis = ns
    elif gammas.size > 1:
        xaxis = gammas
    elif sigmas.size > 1:
        xaxis = sigmas

    xaxis_size = len(xaxis)
    gd_medians = np.empty(xaxis_size)
    lasso_medians = np.empty(xaxis_size)
    oracle_medians = np.empty(xaxis_size)
    gd_err_bars = np.empty((2, xaxis_size))
    lasso_err_bars = np.empty((2, xaxis_size))
    oracle_err_bars = np.empty((2, xaxis_size))

    def transform_id(x):
        return x

    def transform_log(x):
        return np.log(x) / np.log(2)

    if len(ns) == 1:
        transform = transform_id
    else:
        transform = transform_log

    for i, (n, gamma, sigma) in enumerate(itertools.product(ns, gammas, sigmas)):
        # Set up which simulation we corresponds to gamma, sigma and n.
        sp.dataset_size = n
        sp.batch_size = n
        sp.noise_std = sigma
        sp.beta = np.zeros((sp.n_features, 1))
        sp.beta[:sp.k, 0] = np.ones(sp.k) * gamma

        # Extract errors information for the least squares oracle method.
        oracle_errs = simulations[sp]['oracle_ls']['l2_squared_errors'] \
            .squeeze()

        # Choose estimators. For GD we pick using validation data, for the
        # lasso we pick using oracle knowledge of w^{*}.
        gd_opt = find_optimal_gd_values(simulations[sp])['validation_risks']
        lasso_opt = find_optimal_lasso_values(simulations[sp])['oracle_risks']

        # Find medians for the chosen estimators.
        gd_medians[i] = np.median(gd_opt, axis=0)
        lasso_medians[i] = np.median(lasso_opt, axis=0)
        oracle_medians[i] = np.median(oracle_errs)

        # Compute error bars.
        gd_err_bars[0, i] = \
            transform(gd_medians[i]) - transform(np.percentile(gd_opt, 25))
        gd_err_bars[1, i] = \
            transform(np.percentile(gd_opt, 75)) - transform(gd_medians[i])
        lasso_err_bars[0, i] = transform(lasso_medians[i]) \
            - transform(np.percentile(lasso_opt, 25))
        lasso_err_bars[1, i] = transform(np.percentile(lasso_opt, 75)) \
            - transform(lasso_medians[i])
        oracle_err_bars[0, i] = \
            transform(oracle_medians[i]) - \
            transform(np.percentile(oracle_errs, 25))
        oracle_err_bars[1, i] = \
            transform(np.percentile(oracle_errs, 75)) - \
            transform(oracle_medians[i])

    ax.errorbar(transform(xaxis), transform(gd_medians),
                yerr=gd_err_bars, marker='o',
                linewidth=line_width,
                markersize=marker_size)
    ax.errorbar(transform(xaxis), transform(lasso_medians),
                yerr=lasso_err_bars, marker='s',
                linewidth=line_width,
                markersize=marker_size)
    ax.errorbar(transform(xaxis), transform(oracle_medians),
                yerr=oracle_err_bars, marker='^',
                linewidth=line_width,
                markersize=marker_size)
    ax.legend([
        r'gradient descent',
        r'lasso',
        r'least squares oracle'], fontsize=legend_font_size,
        loc='lower left')
    ax.set_xlabel(r'$\log_{2} \,n$')
    ax.set_ylabel(
        r'$\log_{2} \, ||\widehat{\mathbf{w}} - \mathbf{w}^{\star}||^{2}_{2}$')


def plot_everything(simulations_dir):
    simulations = load_simulations_from_directory(simulations_dir)

    fig1, ax1 = get_new_fig_and_ax(1)
    fig2, ax2 = get_new_fig_and_ax(2)

    plot_oracle_errors(simulations, ax1, get_dataset_size_simulation_set_up,
                       mu=0.5)
    ax1.title.set_text(r'Comparing $\ell_{2}$ errors, $\mu = 0.5$')

    plot_oracle_errors(simulations, ax2, get_dataset_size_simulation_set_up,
                       mu=0.0)
    ax2.title.set_text(r'Comparing $\ell_{2}$ errors, $\mu = 0.0$')

    for ax in [ax1, ax2]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(font_size)

    return fig1, fig2


if __name__ == '__main__':
    mpl.use('Agg')
    fig1, fig2 = plot_everything(
        './outputs/gd_vs_lasso_cor/')

    save_fig(fig1, './figures', 'phase_transition_n_cor_0_5')
    save_fig(fig2, './figures', 'phase_transition_n_cor_0_0')
