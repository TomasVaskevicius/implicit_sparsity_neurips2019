import matplotlib as mpl
import numpy as np
import copy

from running.n_vs_k import ns, ks
from plotting.output_loader \
    import load_simulations_from_directory
from plotting.plotting_params import \
    legend_font_size, font_size, get_new_fig_and_ax, \
    marker_size, line_width, save_fig
from plotting.output_utils import \
    find_optimal_gd_values, find_optimal_lasso_values

from mpl_toolkits.axes_grid1 import make_axes_locatable


def set_axis_labels(ax):
    ax.set_ylabel(r'$n$')
    ax.set_xlabel(r'$k$')
    n_labels = list(np.copy(ns)[::-1])
    for i in range(len(n_labels)):
        if i % 2 == 1:
            n_labels[i] = None
    k_labels = list(np.copy(ks))
    for i in range(len(k_labels)):
        if i % 2 == 1:
            k_labels[i] = None
    ax.set_yticks(np.arange(len(ns)))
    ax.set_yticklabels(n_labels)
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels(k_labels)


def compare_l2_errors(ax, fig, simulations):
    sp = list(simulations.keys())[0]
    sp = copy.deepcopy(sp)
    ratios = np.zeros((len(ns), len(ks)))

    for n_id, n in enumerate(ns):
        for k_id, k in enumerate(ks):
            sp.dataset_size = n
            sp.k = k
            sp.beta = np.zeros((sp.n_features, 1))
            sp.beta[:sp.k] = 1
            sp.batch_size = n

            gd_opt = find_optimal_gd_values(simulations[sp])[
                'validation_risks']
            gd_opt = np.median(gd_opt, axis=0)
            lasso_opt = find_optimal_lasso_values(simulations[sp])[
                'oracle_risks']
            lasso_opt = np.median(lasso_opt, axis=0)

            ratios[n_id, k_id] = lasso_opt / gd_opt

    # Reverse the rows order, due to how imshow displays images top to bottom.
    ratios = ratios[::-1, :]
    # Compute log_{2} of the ratio, so that 0 means gd is equal to lasso.
    ratios = np.log(ratios) / np.log(2)

    maximum_val = np.max(np.abs(ratios))
    im = ax.imshow(ratios, cmap='seismic_r', vmax=maximum_val,
                   vmin=-maximum_val)
    set_axis_labels(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(r'$\log_{2} (\operatorname{error\,\, lasso}/\operatorname{error\,\, gd})$',
                       fontsize=font_size)


def plot_l_infty_errors(ax, fig, simulations, algorithm):
    sp = list(simulations.keys())[0]
    sp = copy.deepcopy(sp)
    l_infty_Sc = np.zeros((len(ns), len(ks)))

    for n_id, n in enumerate(ns):
        for k_id, k in enumerate(ks):
            sp.dataset_size = n
            sp.k = k
            sp.beta = np.zeros((sp.n_features, 1))
            sp.beta[:sp.k] = 1
            sp.batch_size = n

            if algorithm == 'gd':
                alg_l_infty_Sc = find_optimal_gd_values(simulations[sp])[
                    'oracle_l_infty_Sc']
            else:
                alg_l_infty_Sc = find_optimal_lasso_values(simulations[sp])[
                    'oracle_l_infty_Sc']
            l_infty_Sc[n_id, k_id] = np.median(alg_l_infty_Sc, axis=0)

    # Reverse the order of rows due to imshow plotting top to bottom.
    l_infty_Sc = l_infty_Sc[::-1, :]
    # Convert to log_{10} scale.
    l_infty_Sc = np.log(l_infty_Sc) / np.log(10)
    im = ax.imshow(l_infty_Sc,
                   cmap='seismic_r',
                   vmax=0,
                   vmin=-12)
    set_axis_labels(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(r'$\log_{10} ||\widehat{\mathbf{w}} \odot \mathbf{1}_{S^{c}}||_{\infty}$',
                       fontsize=font_size)


def plot_everything(simulations_dir):
    simulations = load_simulations_from_directory(simulations_dir)

    fig1, ax1 = get_new_fig_and_ax(1, width=6.0, height=9.0)
    fig2, ax2 = get_new_fig_and_ax(2, width=6.0, height=9.0)
    fig3, ax3 = get_new_fig_and_ax(3, width=6.0, height=9.0)

    compare_l2_errors(ax1, fig1, simulations)
    ax1.title.set_text(r'gd vs lasso $\ell_{2}$ errors')

    plot_l_infty_errors(ax2, fig2, simulations, 'gd')
    ax2.title.set_text(r'gd $\ell_{\infty}$ errors on $S^{c}$')

    plot_l_infty_errors(ax3, fig3, simulations, 'lasso')
    ax3.title.set_text(r'lasso $\ell_{\infty}$ errors on $S^{c}$')

    for ax in [ax1, ax2, ax3]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(font_size)

    return fig1, fig2, fig3


if __name__ == '__main__':
    mpl.use('Agg')
    fig1, fig2, fig3 = plot_everything('./outputs/n_vs_k/')

    save_fig(fig1, './figures', 'n_vs_k_l2_comparisons')
    save_fig(fig2, './figures', 'n_vs_k_l_infty_gd')
    save_fig(fig3, './figures', 'n_vs_k_l_infty_lasso')
