import matplotlib as mpl
import numpy as np
import copy

from plotting.output_loader \
    import load_simulations_from_directory
from plotting.plotting_params import \
    legend_font_size, font_size, get_new_fig_and_ax, \
    marker_size, line_width, save_fig


def get_unique_params(simulations):
    # Only noise_std and alpha will be different between different simulations
    # here. Find the values of alphas and noise_std.
    alphas = []
    noise_stds = []

    for sp in simulations.keys():
        if sp.noise_std not in noise_stds:
            noise_stds.append(sp.noise_std)
        if sp.alpha not in alphas:
            alphas.append(sp.alpha)

    alphas.sort()
    noise_stds.sort()
    alphas = np.array(alphas)[::-1]
    noise_stds = np.array(noise_stds)
    return alphas, noise_stds


def plot_errors_path(sim, maxt, ax, **kwargs):
    errs = sim['gd_performance']['l2_squared_errors'].squeeze()
    medians = np.log(np.median(errs, axis=0)) / np.log(2)
    lower_percentile = \
        medians - np.log(np.percentile(errs, 25, axis=0)) / np.log(2)
    upper_percentile = \
        np.log(np.percentile(errs, 75, axis=0)) / np.log(2) - medians
    bars = np.stack((lower_percentile, upper_percentile))
    ax.errorbar(
        sim['simulation_parameters'].observers_frequency
        * (np.arange(maxt)),
        medians[:maxt],
        yerr=bars[:, :maxt], **kwargs)


def plot_alpha_effect(ax):
    simulations = load_simulations_from_directory(
        './outputs/alpha_effect/alphas/')

    alphas, noise_stds = get_unique_params(simulations)
    sp = copy.deepcopy(list(simulations.keys())[0])
    sp.noise_std = noise_stds[0]

    maxt = 21
    markers = ['o', 's', '^']
    for alpha, marker in zip(alphas, markers):
        sp.alpha = alpha
        plot_errors_path(
            simulations[sp],
            maxt,
            ax,
            marker=marker,
            markersize=marker_size,
            linewidth=line_width)
    ax.legend([
        r'$\alpha=10^{-2}$',
        r'$\alpha=10^{-3}$',
        r'$\alpha=10^{-4}$'], fontsize=legend_font_size, loc='upper right')
    ax.title.set_text(r'Effect of initiliaztion size $\alpha$')
    ax.set_xlabel(r'Number of iterations $t$')
    ax.set_ylabel(
        r'$\log_{2} ||\mathbf{w}_{t} - \mathbf{w}^{\star}||_{2}^{2}$')


def plot_coordinates_path_t(params, k, ax):
    # Plot the noise variables.
    # Since we plot 10 000 noise lines, rasterized image becomes too large.
    # Instead, we will plot 1000 noise lines (with maximum noise) which does
    # not change things visually.
    n_noise_lines = 1000
    abs_noises = np.absolute(params[:, k:])
    max_abs_noises = np.max(abs_noises, axis=0)
    args_to_plot = np.argsort(max_abs_noises)[-n_noise_lines:] + k
    for i in range(params.shape[1]):
        if i not in args_to_plot:
            continue
        line_noise, = ax.plot(
            np.arange(params.shape[0]),
            params[:, i],
            c='C3', linestyle='dotted', linewidth=line_width,
            rasterized=True)

    # Plot the true variables.
    for i in range(k):
        line_signal, = ax.plot(
            np.arange(params.shape[0]),
            params[:, i], c='C0', linewidth=line_width)

    ax.hlines(1.0, 0, params.shape[0],
              colors='C2', linestyles='dashed', linewidth=line_width)

    ax.legend(
        (line_signal, line_noise),
        (r'$i \in S$', r'$i \in S^{c}$'),
        fontsize=legend_font_size, loc='upper right')


def plot_coordinates_paths(ax1, ax2):
    simulations = load_simulations_from_directory(
        './outputs/alpha_effect/paths/')

    alphas, noise_stds = get_unique_params(simulations)
    sp = copy.deepcopy(list(simulations.keys())[0])
    sp.noise_std = noise_stds[0]

    # We will remove the first two iterations since they change initialization
    # size and step size and everything by the hyperparameter tuning obsever,
    # even if it is not used later on.
    maxts = [102, 302]
    axes = [ax1, ax2]
    for ax, alpha, maxt in zip(axes, alphas, maxts):
        sp.alpha = alpha
        params = simulations[sp]['params']['params']
        plot_coordinates_path_t(params[2:maxt], sp.k, ax)
        ax.set_xlabel(r'Number of iterations $t$')
        ax.set_ylabel(r'$w_{t,i}$')

    ax1.title.set_text(r'Coordinates paths with $\alpha = 10^{-3}$')
    ax2.title.set_text(r'Coordinates paths with $\alpha = 10^{-12}$')
    bottom1, top1 = ax1.get_ylim()
    bottom2, top2 = ax2.get_ylim()
    bottom = min(bottom1, bottom2)
    top = max(top1, top2)
    # To make scaling the same as for plots in gd_vs_lasso
    # we override the top.
    top = 1.5
    ax1.set_ylim(bottom, top)
    ax2.set_ylim(bottom, top)


def plot_everything():
    fig1, ax1 = get_new_fig_and_ax(1)
    fig2, ax2 = get_new_fig_and_ax(2)
    fig3, ax3 = get_new_fig_and_ax(3)

    plot_alpha_effect(ax1)
    plot_coordinates_paths(ax2, ax3)
    for ax in [ax1, ax2, ax3]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(font_size)

    return (fig1, fig2, fig3)


if __name__ == '__main__':
    mpl.use('Agg')
    fig1, fig2, fig3 = plot_everything()

    save_fig(fig1, './figures', 'alpha_effects')
    save_fig(fig2, './figures', 'alpha_large')
    save_fig(fig3, './figures', 'alpha_small')
