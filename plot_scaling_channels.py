import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {'size': 14}
matplotlib.rc('font', **font)


def aggregate_timing(times, aggregate_method='mean'):
    if aggregate_method == 'mean':
        return np.mean(times)
    elif aggregate_method == 'median':
        return np.median(times)
    elif aggregate_method == 'max':
        return np.max(times)
    else:
        raise ValueError('unknown aggregate_time: {}'.format(aggregate_method))


def plot_convergence(all_results_df, threshold, aggregate_method, save_name):
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
                fig = plt.figure(figsize=(12, 9))
                ax = fig.gca()
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                timing_z_update = []
                timing_d_update = []
                timing_full_update = []
                for n_channels in span_channels:
                    # aggregate all runs (different random states)
                    this_res_2 = this_res[this_res['n_channels'] == n_channels]
                    timing_z, timing_d, timing_full = [], [], []
                    for times in this_res_2['times'].values:
                        timing_z.extend(times[1::2])
                        timing_d.extend(times[2::2])
                        timing_full.append(np.sum(times))
                    timing_z_update.append(aggregate_timing(timing_z,
                                                            aggregate_method))
                    print(timing_d)
                    timing_d_update.append(aggregate_timing(timing_d,
                                                            aggregate_method))
                    timing_full_update.append(np.mean(timing_full))

                ax.set_yscale('log')
                plt.plot(span_channels, timing_z_update, label="z_update")
                plt.plot(span_channels, timing_d_update, label="d_update")
                plt.plot(span_channels, timing_full_update, label="full_update")

                plt.xlabel('n_channels')
                plt.ylabel('Time (s)')
                plt.legend()
                # plt.ylim(ymin=ymin / 10)
                # plt.title('K = %d, L = %d' % (n_atoms, n_times_atom))
                plt.gca().tick_params(axis='x', which='both', bottom='off',
                                      top='off')
                plt.gca().tick_params(axis='y', which='both', left='off',
                                      right='off')
                plt.grid(True)
                plt.tight_layout()

                fig.savefig(save_name + '_bench_%s_K%d_L%d.png' %
                            (label, n_atoms, n_times_atom), dpi=150)


##############################################################################
# load the results from file

load_name = 'methods_scaling.pkl'

load_name = os.path.join('figures', load_name)
all_results_df = pd.read_pickle(load_name)

if 'threshold' not in globals():
    threshold = (all_results_df['stopping_pobj'] / all_results_df['best_pobj']
                 - 1).unique()
    assert threshold.size == 1
    threshold = threshold[0]

# force threshold
threshold = 0.001
normalize_method = None
save_name = load_name[:-4]

for aggregate_method in ['mean', 'median', 'max']:
    plot_convergence(all_results_df, threshold, aggregate_method, save_name)

# plt.show()
plt.close('all')
