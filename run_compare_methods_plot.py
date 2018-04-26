import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {'size': 14}
matplotlib.rc('font', **font)


def normalize_pobj(pobj, best_pobj, normalize_method='best'):
    if normalize_method == 'best':
        pobj = (pobj - best_pobj) / best_pobj
    elif normalize_method == 'last':
        pobj = [(p - p.min()) / p.min() for p in pobj]
    elif normalize_method in [None, 'short']:
        pass
    else:
        raise ValueError('unknown normalize_method: %s' % normalize_method)

    return pobj


def plot_convergence(all_results_df, threshold, normalize_method, save_name):
    save_name += '_%s' % (normalize_method, )
    labels = all_results_df['label'].unique()
    if 'M-step ' in labels:
        labels[3] = labels[2]
        labels[2] = 'M-step '
    for n_atoms in all_results_df['n_atoms'].unique():
        for n_times_atom in all_results_df['n_times_atom'].unique():
            this_res = all_results_df
            this_res = this_res[this_res['n_atoms'] == n_atoms]
            this_res = this_res[this_res['n_times_atom'] == n_times_atom]

            if this_res.size == 0:
                continue

            best_pobj = min(this_res['best_pobj'].unique())

            # draw a different figure for each setting
            fig = plt.figure(figsize=(12, 9))
            ax = fig.gca()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # ymin = np.inf
            tmax = []
            for label in labels:
                # aggregate all runs (different random states)
                this_res_2 = this_res[this_res['label'] == label]
                times = this_res_2['times']
                pobj = this_res_2['pobj']

                pobj = normalize_pobj(pobj, best_pobj, normalize_method)
                if normalize_method in [None, 'short']:
                    plot_func = plt.plot
                else:
                    plot_func = plt.loglog

                if True:
                    # geometric mean on the n_iter_min first iterations
                    n_iter_min = min([t.shape[0] for t in pobj])
                    pobj_stack = np.vstack([p[:n_iter_min] for p in pobj])
                    pobj_stack = np.log10(pobj_stack + 1e-15)
                    pobj_mean = 10 ** (np.mean(pobj_stack, axis=0))
                    times_mean = np.vstack([t[:n_iter_min] for t in times])
                    times_mean = times_mean.mean(axis=0)
                    tmax.append(times_mean[-1])

                    if normalize_method == 'last' and True:
                        last = np.where(pobj_mean <= threshold / 10)[0][0]
                        times_mean = times_mean[:last]
                        pobj_mean = pobj_mean[:last]

                    # plot the mean
                    plot_func(times_mean, pobj_mean, '.-', label=label,
                              alpha=1.)
                    color = ax.lines[-1].get_color()
                    # for times_, pobj_ in zip(times, pobj):
                    #     plt.semilogy(times_, pobj_, alpha=0.2, color=color)

                    # ymin = min(ymin, pobj_mean[pobj_mean > 0].min())

                else:
                    color = None  # new color for new label
                    for times_, pobj_ in zip(times, pobj):
                        if pobj_[-1] <= threshold:
                            if normalize_method == 'last' and True:
                                last = np.where(pobj_ <= threshold / 10)[0][0]
                                times_ = times_[:last]
                                pobj_ = pobj_[:last]
                            marker, alpha = None, 1.0

                            plot_func(times_, pobj_, marker=marker,
                                      label=label, alpha=alpha, color=color)
                            # reuse color for next lines
                            color = ax.lines[-1].get_color()
                            # don't duplicate label for next lines
                            label = None

                    for times_, pobj_ in zip(times, pobj):
                        if pobj_[-1] > threshold:
                            marker, alpha = None, 0.3

                            plot_func(times_, pobj_, marker=marker,
                                      label=label, alpha=alpha, color=color)
                            # reuse color for next lines
                            color = ax.lines[-1].get_color()
                            # don't duplicate label for next lines
                            label = None

            plt.xlabel('Time (s)')
            if normalize_method in [None, 'short']:
                plt.ylabel('objective')
                if normalize_method == 'short':
                    xmax = np.sort(tmax)[0] / 10
                    plt.xlim(-xmax / 10, xmax)
            elif normalize_method == 'last':
                plt.ylabel('(objective_i - best_i) / best_i')
            else:
                plt.ylabel('(objective - best) / best')
            plt.legend(loc=0, ncol=1)
            # plt.ylim(ymin=ymin / 10)
            # plt.title('K = %d, L = %d' % (n_atoms, n_times_atom))
            plt.gca().tick_params(axis='x', which='both', bottom='off',
                                  top='off')
            plt.gca().tick_params(axis='y', which='both', left='off',
                                  right='off')
            plt.grid(True)
            plt.tight_layout()

            fig.savefig(save_name + '_bench_K%d_L%d.png' %
                        (n_atoms, n_times_atom), dpi=150)


##############################################################################
# load the results from file

load_name = 'methods_.pkl'

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

for normalize_method in [None, 'short', 'best', 'last']:
    plot_convergence(all_results_df, threshold, normalize_method, save_name)

# plt.show()
plt.close('all')
