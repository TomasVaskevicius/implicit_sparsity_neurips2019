import matplotlib as mpl
import numpy as np
import copy

from plotting.output_loader \
    import load_simulations_from_directory
from plotting.plotting_params import \
    legend_font_size, font_size, get_new_fig_and_ax, \
    marker_size, line_width, save_fig


def plot_errors_path(errs, xaxis, ax, **kwargs):
    means = np.median(errs, axis=0)
    lower_percentile = np.percentile(errs, 25, axis=0)
    upper_percentile = np.percentile(errs, 75, axis=0)

    means = np.log(means) / np.log(2)
    lower_percentile = np.log(lower_percentile) / np.log(2)
    upper_percentile = np.log(upper_percentile) / np.log(2)
    bars = np.stack((means - lower_percentile, upper_percentile - means))

    markers, caps, bars = ax.errorbar(
        xaxis,
        means[:],
        yerr=bars, **kwargs)
    [bar.set_alpha(0.15) for bar in bars]

    ax.set_xlabel(r'Number of iterations $t$')
    ax.set_ylabel(
        r'$\log_{2} ||\mathbf{w}_{t} - \mathbf{w}^{\star}||_{2}^{2}$')


def compare_l2_errors(ax1, ax2):
    simulations = load_simulations_from_directory(
        './outputs/exponential_convergence/')

    # Load constant step size simulations.
    sp = copy.deepcopy(list(simulations.keys())[0])
    sim = simulations[sp]
    errs = sim['gd_performance']['l2_squared_errors'].squeeze()
    xaxis = np.arange(errs.shape[1]) * sp.observers_frequency
    plot_errors_path(errs, xaxis, ax2, linewidth=2)
    # Plot vertical lines where step sizes are doubled.
    doubling_interval = (640 // sp.exponentiation_rate) \
        * np.ceil(np.log(1.0 / sp.alpha))
    interval_time = 1
    bottom, top = ax2.get_ylim()
    while doubling_interval * interval_time <= xaxis[-1]:
        ax2.vlines(doubling_interval * interval_time,
                   bottom,
                   top,
                   linestyles='dotted',
                   colors='black')
        interval_time += 1

    # Load exponential step size simulations.
    sp = copy.deepcopy(list(simulations.keys())[1])
    sim = simulations[sp]
    errs = sim['gd_performance']['l2_squared_errors'].squeeze()
    # Make the number of observations the same for both plots.
    errs = errs[:, ::16]
    xaxis = np.arange(errs.shape[1]) * sp.observers_frequency * 16
    plot_errors_path(errs, xaxis, ax1, linewidth=line_width)


def plot_everything():
    fig1, ax1 = get_new_fig_and_ax(1)
    fig2, ax2 = get_new_fig_and_ax(2)

    compare_l2_errors(ax1, ax2)
    bottom1, top1 = ax1.get_ylim()
    bottom2, top2 = ax2.get_ylim()
    bottom = min(bottom1, bottom2)
    top = max(top1, top2)
    ax1.set_ylim(bottom, top)
    ax2.set_ylim(bottom, top)
    ax1.title.set_text('Constant step size')
    ax2.title.set_text('Increasing step sizes')

    for ax in [ax1, ax2]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(font_size)

    return fig1, fig2


if __name__ == '__main__':
    mpl.use('Agg')
    fig1, fig2 = plot_everything()

    save_fig(fig1, './figures', 'constant_step_size')
    save_fig(fig2, './figures', 'increasing_step_sizes')
