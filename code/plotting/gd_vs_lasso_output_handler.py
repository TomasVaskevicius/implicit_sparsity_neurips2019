import matplotlib as mpl
import numpy as np
import copy
import itertools

from running.gd_vs_lasso \
    import get_signal_size_simulation_set_up, \
    get_sigma_simulation_set_up, \
    get_dataset_size_simulation_set_up
from plotting.output_loader \
    import load_simulations_from_directory
from plotting.output_utils import \
    find_optimal_gd_values, find_optimal_lasso_values
from plotting.plotting_params import \
    legend_font_size, font_size, get_new_fig_and_ax, \
    marker_size, line_width, save_fig


# For plotting parameter paths against the l1 norm.
def plot_coordinates_path(params, k, ax, max_l1_norm=100):
    l1_norms = np.empty(params.shape[0])
    for i in range(params.shape[0]):
        l1_norms[i] = np.sum(np.absolute(params[i, :]))
    mask = l1_norms <= max_l1_norm

    # Plot the noise variables.
    # Since we plot 10 000 noise lines, rasterized image becomes too large.
    # Instead, we will plot 1000 noise lines (with maximum noise) which does
    # not change things visually.
    n_noise_lines = 1000
    abs_noises = np.absolute(params[mask, k:])
    max_abs_noises = np.max(abs_noises, axis=0)
    args_to_plot = np.argsort(max_abs_noises)[-n_noise_lines:] + k
    for i in range(params.shape[1]):
        if i not in args_to_plot:
            continue
        line_noise, = ax.plot(
            l1_norms[mask],
            params[mask, i],
            c='C3', linestyle='dotted', linewidth=line_width - 1,
            rasterized=True)

    # Plot the true variables.
    for i in range(k):
        line_signal, = ax.plot(
            l1_norms[mask],
            params[mask, i], c='C0', linewidth=line_width - 1)

    ax.legend(
        (line_signal, line_noise),
        (r'$i \in S$', r'$i \in S^{c}$'),
        fontsize=legend_font_size,
        loc='upper right')


def plot_coordinates_paths(simulations, ax1, ax2):
    sp = list(simulations.keys())[0]
    sp = copy.deepcopy(sp)
    sp.beta[:sp.k, 0] = np.ones(sp.k) * 1.0
    sp.dataset_size = 500
    sp.batch_size = 500
    sp.noise_std = 1.0
    sp.observe_parameters = 1
    sp.store_glmnet_path = 1

    gd_params = \
        simulations[sp]['params']['params'][:, :]
    lasso_params = \
        simulations[sp]['lasso_performance']['coef_paths'][0, :, :]

    # Truncate both plots at max l1 norm of one of the methods.
    max_l1_norm = min(
        np.sum(np.absolute(gd_params[-1, :])),
        np.sum(np.absolute(lasso_params[0, :])))

    plot_coordinates_path(gd_params, sp.k, ax1, max_l1_norm)
    plot_coordinates_path(lasso_params, sp.k, ax2, max_l1_norm)
    ax1.hlines(sp.beta[0, 0], 0, max_l1_norm,
               colors='C2', linestyles='dashed', linewidth=line_width - 1)
    ax2.hlines(sp.beta[0, 0], 0, max_l1_norm, colors='C2',
               linestyles='dashed', linewidth=line_width - 1)
    ax1.set_ylabel(r'$w_{t,i}$')
    ax2.set_ylabel(r'$w_{t,i}$')
    ax1.set_xlabel(r'$||\mathbf{w}_{t}||_{1}$')
    ax2.set_xlabel(r'$||\mathbf{w}_{\lambda}||_{1}$')
    ax1.title.set_text(r'Gradient descent')
    ax2.title.set_text(r'Lasso')
    bottom1, top1 = ax1.get_ylim()
    bottom2, top2 = ax2.get_ylim()
    bottom = min(bottom1, bottom2)
    top = max(top1, top2)
    ax1.set_ylim(bottom, top)
    ax2.set_ylim(bottom, top)


def plot_oracle_errors(simulations, ax, simulation_set_up):
    sp = list(simulations.keys())[0]
    sp = copy.deepcopy(sp)

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
        r'least squares oracle'], fontsize=legend_font_size)
    ax.title.set_text(r'Comparing $\ell_{2}$ errors')


def plot_everything():
    simulations = load_simulations_from_directory(
        './outputs/gd_vs_lasso/')

    fig1, ax1 = get_new_fig_and_ax(1)
    fig2, ax2 = get_new_fig_and_ax(2)
    fig3, ax3 = get_new_fig_and_ax(3)
    fig4, ax4 = get_new_fig_and_ax(4)
    fig5, ax5 = get_new_fig_and_ax(5)

    plot_oracle_errors(simulations, ax1, get_signal_size_simulation_set_up)
    plot_oracle_errors(simulations, ax2, get_sigma_simulation_set_up)
    plot_oracle_errors(simulations, ax3, get_dataset_size_simulation_set_up)
    ax1.set_xlabel(r'$\gamma$')
    ax1.set_ylabel(r'$||\widehat{\mathbf{w}} - \mathbf{w}^{\star}||^{2}_{2}$')
    ax2.set_xlabel(r'$\sigma$')
    ax2.set_ylabel(r'$||\widehat{\mathbf{w}} - \mathbf{w}^{\star}||^{2}_{2}$')
    ax3.set_xlabel(r'$\log_{2} \,n$')
    ax3.set_ylabel(
        r'$\log_{2} \, ||\widehat{\mathbf{w}} - \mathbf{w}^{\star}||^{2}_{2}$')

    # Here we solve the phase transition equation gamma = 2 * maxnoise
    # for each simulation setting.
    d = list(simulations.keys())[0].n_features

    ns, gammas, sigmas = get_signal_size_simulation_set_up()
    transition_gamma = 2 * sigmas[0] * np.sqrt(2 * np.log(2 * d)) \
        / np.sqrt(ns[0])

    ns, gammas, sigmas = get_sigma_simulation_set_up()
    transition_sigma = 0.5 * np.sqrt(ns[0]) * gammas[0] \
        / np.sqrt(2 * np.log(2 * d))

    ns, gammas, sigmas = get_dataset_size_simulation_set_up()
    transition_n = 4 * sigmas[0]**2 * 2 * np.log(2 * d) / gammas[0]**2
    transition_log2_n = np.log(transition_n) / np.log(2)

    transitions = [transition_gamma, transition_sigma, transition_log2_n]
    axes = [ax1, ax2, ax3]
    for transition, ax in zip(transitions, axes):
        bottom, top = ax.get_ylim()
        ax.vlines(transition,
                  bottom,
                  top,
                  linestyles='solid',
                  colors='red',
                  linewidth=line_width)

    plot_coordinates_paths(simulations, ax4, ax5)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(font_size)

    return (fig1, fig2, fig3, fig4, fig5)


if __name__ == '__main__':
    mpl.use('Agg')
    fig1, fig2, fig3, fig4, fig5 = plot_everything()

    save_fig(fig1, './figures', 'phase_transition_gamma')
    save_fig(fig2, './figures', 'phase_transition_sigma')
    save_fig(fig3, './figures', 'phase_transition_n')
    save_fig(fig4, './figures', 'gd_path')
    save_fig(fig5, './figures', 'lasso_path')
