import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fontsize = 14
font = {'size': 14}
matplotlib.rc('font', **font)


def aggregate_timing(times, aggregate_method='mean'):
    if aggregate_method == 'mean':
        return np.mean(times), np.std(times)
    elif aggregate_method == 'median':
        return np.median(times), np.std(times)
    elif aggregate_method == 'max':
        return np.max(times), np.std(times)
    else:
        raise ValueError('unknown aggregate_time: {}'.format(aggregate_method))


def plot_scaling_channels(all_results_df, aggregate_method, save_name):
    save_name += '_{}'.format(aggregate_method)
    span_channels = all_results_df['n_channels'].unique()
    for n_atoms in all_results_df['n_atoms'].unique():
        for n_times_atom in all_results_df['n_times_atom'].unique():
            for label in all_results_df['label'].unique():
                this_res = all_results_df
                this_res = this_res[this_res['n_atoms'] == n_atoms]
                this_res = this_res[this_res['n_times_atom'] == n_times_atom]
                this_res = this_res[this_res['label'] == label]

                if this_res.size == 0:
                    continue

                # draw a different figure for each setting
                fig = plt.figure(figsize=(6, 4))
                ax = fig.gca()
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

                    z_update.append(aggregate_timing(timing_z,
                                                     aggregate_method))
                    d_update.append(aggregate_timing(timing_d,
                                                     aggregate_method))
                    full_update.append(aggregate_timing(timing_full,
                                                        aggregate_method))

                z_update = np.array(z_update) / z_update[0]
                d_update = np.array(d_update) / d_update[0]
                full_update = np.array(full_update) / full_update[0]

                # ax.set_yscale('log')
                plots = [
                    (z_update, "C0", "z step"),
                    (d_update, "C1", "d step"),
                    (full_update, "C2", "z+d steps")
                ]
                for timing, color, name in plots:
                    plt.plot(span_channels, timing[:, 0], c=color, label=name)
                    # m, std = timing[:, 0], timing[:, 1]
                    # plt.fill_between(span_channels, m - std, m + std,
                    #                  color=color, alpha=.1)

                # Plot first diagonal
                # t = np.arange(101)
                # plt.plot(t, t, "k--")
                # plt.text(23, 46, "linear scaling", rotation=60, fontsize=14,
                #          bbox=dict(facecolor="white", edgecolor="white"))

                plt.xlabel('# of channels P', fontsize=fontsize)
                plt.ylabel('$^{time_{P}}/_{time_1}$', fontsize=22)
                plt.legend(frameon=True, fontsize=14, ncol=3, columnspacing=.5)
                plt.gca().tick_params(axis='x', which='both', bottom=False,
                                      top=False)
                plt.gca().tick_params(axis='y', which='both', left=False,
                                      right=False)
                plt.gca().ticklabel_format(style="plain", axis='y')
                plt.xticks([1, 50, 100, 150, 200])
                plt.yticks([1, 2, 3, 4])
                plt.ylim((1, 4))
                plt.xlim((1, span_channels.max()))
                plt.grid(True)
                plt.pause(.001)
                plt.tight_layout()

                fig.savefig(save_name + '_%s_K%d_L%d.png' %
                            (label, n_atoms, n_times_atom), dpi=150)


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
        'Plot the scaling of multichannel CSC relatively to the number of channels P.')
    parser.add_argument('--fname', type=str, default='figures/methods_scaling_reg0.005.pkl',
                        help='Name of the file to plot from.')
    args = parser.parse_args()

    ##############################################################################
    # load the results from file

    load_name = args.fname

    all_results_df = pd.read_pickle(load_name)

    normalize_method = None
    save_name = load_name.replace("methods_scaling", "scaling_channels").replace(".pkl", '')
    save_name = save_name.replace(".", "_")

    for aggregate_method in ['mean', 'median', 'max']:
        plot_scaling_channels(all_results_df, aggregate_method, save_name)

    # plt.show()
    plt.close('all')
