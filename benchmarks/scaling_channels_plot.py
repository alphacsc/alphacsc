"""Benchmark the scaling of alphacsc algorithm with multiple channels.

This script requires `pandas` which can be installed with `pip install pandas`.

This script plots the results saved by the script scaling_channels_run.py,
which should be run beforehand.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fontsize = 14
font = {'size': fontsize}
matplotlib.rc('font', **font)

figsize = (6, 3.4)


def color_palette(n_colors=4, cmap='viridis', extrema=False):
    if extrema:
        bins = np.linspace(0, 1, n_colors)
    else:
        bins = np.linspace(0, 1, n_colors * 2 - 1 + 2)[1:-1:2]

    cmap = plt.get_cmap(cmap)
    palette = list(map(tuple, cmap(bins)[:, :3]))
    return palette


def aggregate_timing(times, aggregate_method='mean'):
    if aggregate_method == 'mean':
        return np.mean(times), np.std(times)
    elif aggregate_method == 'median':
        return np.median(times), np.std(times)
    elif aggregate_method == 'max':
        return np.max(times), np.std(times)
    else:
        raise ValueError('unknown aggregate_time: {}'.format(aggregate_method))


def plot_scaling_channels(all_results_df, aggregate_method, save_name,
                          wohlberg_df, extension):
    save_name += '_{}'.format(aggregate_method)
    span_channels = all_results_df['n_channels'].unique()
    label = all_results_df['label'].unique()[0]
    n_times_atom = all_results_df['n_times_atom'].unique()[0]
    n_atoms = all_results_df['n_atoms'].unique()[0]

    this_res = all_results_df

    if this_res.size == 0:
        return

    # draw a different figure for each setting
    fig = plt.figure(figsize=figsize)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    z_update, d_update, full_update = [], [], []
    for n_channels in span_channels:
        # aggregate all runs (different random states)
        this_res_2 = this_res[this_res['n_channels'] == n_channels]
        timing_z, timing_d, timing_full = [], [], []
        for times in this_res_2['times'].values:
            timing_z.extend(times[1::2])
            timing_d.extend(times[2::2])
            timing_full.extend(times[1::2] + times[2::2])

        z_update.append(aggregate_timing(timing_z, aggregate_method))
        d_update.append(aggregate_timing(timing_d, aggregate_method))
        full_update.append(aggregate_timing(timing_full, aggregate_method))

    z_update = np.array(z_update) / z_update[0]
    d_update = np.array(d_update) / d_update[0]
    full_update = np.array(full_update) / full_update[0]

    # ax.set_yscale('log')
    plots = [
        (z_update, "z step"),
        (d_update, "d step"),
        (full_update, "z+d steps")
    ]
    # colors = color_palette(len(plots))[::-1]
    colors = ['C0', 'C1', 'C2']
    for (timing, name), color in zip(plots, colors):
        plt.plot(span_channels, timing[:, 0], c=color, label=name)

    # Plot first diagonal
    if wohlberg_df is not None:
        w_channels = wohlberg_df['n_channels'].unique()
        wohlberg_curve = []
        for n_channels in w_channels:
            # aggregate all runs (different random states)
            this_res = wohlberg_df[wohlberg_df['n_channels'] == n_channels]
            wohlberg_timing = [], [], []
            for times in this_res['times'].values:
                wohlberg_timing.extend(times[1::2] + times[2::2])

            wohlberg_curve.append(aggregate_timing(wohlberg_timing,
                                                   aggregate_method))
        plt.plot(wohlberg_curve, "k--", label="Wohlberg (2017)")
    # t = np.arange(101)
    # plt.plot(t, t, "k--")
    # plt.text(23, 46, "linear scaling", rotation=60, fontsize=14,
    #          bbox=dict(facecolor="white", edgecolor="white"))

    plt.xlabel('# of channels P', fontsize=fontsize)
    plt.ylabel('$^{time_{P}}/_{time_1}$', fontsize=20)
    plt.legend(frameon=True, fontsize=14, ncol=3, columnspacing=.5)
    plt.gca().tick_params(axis='x', which='both', bottom=False, top=False)
    plt.gca().tick_params(axis='y', which='both', left=False, right=False)
    plt.gca().ticklabel_format(style="plain", axis='y')
    plt.xticks([1, 50, 100, 150, 200])
    # plt.yticks([1, 2, 3, 4])
    # plt.ylim((1, 4))
    plt.xlim((1, span_channels.max()))
    plt.grid(True)
    plt.pause(.001)
    plt.tight_layout()

    fig.savefig(save_name + '_{}_K{}_L{}.{}'.format(
        label, n_atoms, n_times_atom, extension), dpi=150)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        'Plot the scaling of multichannel CSC relatively to the number of '
        'channels P.')
    parser.add_argument('--fname', type=str,
                        default='figures/methods_scaling_reg0.005.pkl',
                        help='Name of the file to plot from.')
    parser.add_argument('--pdf', action='store_true',
                        help='Output pdf figures for final form.')
    parser.add_argument('--wohlberg', action='store_true',
                        help='Output pdf figures for final form.')
    args = parser.parse_args()

    ###############################################################
    # load the results from file

    load_name = args.fname
    extension = "png"
    if args.pdf:
        extension = "pdf"

    all_results_df = pd.read_pickle(load_name)

    normalize_method = None
    save_name = load_name.replace("methods_scaling",
                                  "scaling_channels").replace(".pkl", '')
    save_name = save_name.replace(".", "_")

    wohlberg_df = None
    if args.wohlberg:
        wohlberg_name = load_name.replace(".pkl", "_wohlberg.pkl")
        wohlberg_df = pd.read_pickle(wohlberg_name)

    for aggregate_method in ['mean', 'median', 'max']:
        plot_scaling_channels(all_results_df, aggregate_method, save_name,
                              wohlberg_df=wohlberg_df, extension=extension)

    # plt.show()
    plt.close('all')
