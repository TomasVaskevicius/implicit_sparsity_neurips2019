import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle

legend_font_size = 24
font_size = 38
marker_size = 8
line_width = 3

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['legend.framealpha'] = 0.0
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24


def get_new_fig_and_ax(fig_id, width=9.0, height=6.0):
    fig = plt.figure(fig_id)
    ax = fig.gca()
    fig.set_size_inches(width, height)
    fig.tight_layout(pad=1.5 * float(font_size)/10.0)
    return fig, ax


save_fig_extensions = ['png', 'pdf']


def create_directories(path):
    os.makedirs(path, exist_ok=True)
    for extension in save_fig_extensions:
        os.makedirs(path + '/' + extension, exist_ok=True)


def save_fig(fig, path, fname, **kwargs):
    create_directories(path)
    for extension in save_fig_extensions:
        full_path = \
            path + '/' + extension + '/' + fname + '.' + extension
        if extension != 'pickle':
            fig.savefig(full_path, dpi=300, transparent=True,
                        bbox_inches='tight', **kwargs)
        else:
            with open(full_path, 'wb') as _file:
                pickle.dump(fig, _file)
