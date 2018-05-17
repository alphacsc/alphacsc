import os
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rc('font', size=18)
matplotlib.rc('mathtext', fontset='cm')


def color_palette(n_colors=4, cmap='viridis', extrema=False):
    if extrema:
        bins = np.linspace(0, 1, n_colors)
    else:
        bins = np.linspace(0, 1, n_colors * 2 - 1 + 2)[1:-1:2]

    cmap = plt.get_cmap(cmap)
    palette = list(map(tuple, cmap(bins)[:, :3]))
    return palette


def normalize_pobj(pobj, best_pobj=None, normalize_method='best'):
    if normalize_method == 'best':
        assert best_pobj is not None
        pobj = (pobj - best_pobj) / best_pobj
    elif normalize_method == 'last':
        pobj = [(p - p.min()) / p.min() for p in pobj]
        # pobj = [p / p[0] if p[0] != 0 else p for p in pobj]
    elif normalize_method == 'diff':
        pobj = [-np.diff(p) for p in pobj]
    elif normalize_method in [None, 'short']:
        pass
    else:
        raise ValueError('unknown normalize_method: %s' % normalize_method)

    return pobj


def plot_convergence(data_frame, threshold, normalize_method, save_name):
    save_name += '_%s' % (normalize_method, )
    for n_atoms in data_frame['n_atoms'].unique():

        for n_times_atom in data_frame['n_times_atom'].unique():
            this_res = data_frame
            this_res = this_res[this_res['n_atoms'] == n_atoms]
            this_res = this_res[this_res['n_times_atom'] == n_times_atom]
            labels = this_res['label'].unique()

            if this_res.size == 0:
                continue

            best_pobj = min([min(pobj) for pobj in this_res['pobj']])

            # draw a different figure for each setting
            fig = plt.figure(figsize=(6, 4))
            ax = fig.gca()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # ymin = np.inf
            tmax = []
            colors = color_palette(len(labels))
            for label, color in zip(labels, colors):
                if label == 'find_best_pobj':
                    continue
                # aggregate all runs (different random states)
                this_res_2 = this_res[this_res['label'] == label]
                times = this_res_2['times']
                pobj = this_res_2['pobj']

                pobj = normalize_pobj(pobj, best_pobj, normalize_method)

                if True:
                    # geometric mean on the n_iter_min first iterations
                    n_iter_min = min([t.shape[0] for t in pobj])
                    pobj_stack = np.vstack([p[:n_iter_min] for p in pobj])
                    pobj_stack = np.log10(pobj_stack + 1e-15)
                    pobj_mean = 10 ** (np.mean(pobj_stack, axis=0))
                    times_mean = np.vstack([t[:n_iter_min] for t in times])
                    times_mean = times_mean.mean(axis=0)
                    if times_mean.size == 0:
                        continue
                    tmax.append(times_mean[-1])

                    times_mean = times_mean[1:]
                    pobj_mean = pobj_mean[1:]

                    if normalize_method == 'last' and True:
                        last = np.where(pobj_mean <= threshold)[0]
                        if last.size == 0:
                            continue
                        last = last[0]
                        times_mean = times_mean[:last]
                        pobj_mean = pobj_mean[:last]

                    style = '--' if 'Proposed' in label else '-'

                    # plot the mean
                    plt.plot(times_mean, pobj_mean, style, label=label,
                             alpha=1., markevery=0.1, lw=2., color=color)
                    # color = ax.lines[-1].get_color()
                    # for times_, pobj_ in zip(times, pobj):
                    #     plt.semilogy(times_, pobj_, alpha=0.2, color=color)

                    # ymin = min(ymin, pobj_mean[pobj_mean > 0].min())

                else:
                    # color = None  # new color for new label
                    for times_, pobj_ in zip(times, pobj):
                        if pobj_[-1] <= threshold:
                            if normalize_method == 'last' and True:
                                last = np.where(pobj_ <= threshold)[0][0]
                                times_ = times_[:last]
                                pobj_ = pobj_[:last]
                            marker, alpha = None, 1.0

                            plt.plot(times_, pobj_, marker=marker, label=label,
                                     alpha=alpha, color=color)
                            # reuse color for next lines
                            # color = ax.lines[-1].get_color()
                            # don't duplicate label for next lines
                            label = None

                    for times_, pobj_ in zip(times, pobj):
                        if pobj_[-1] > threshold:
                            marker, alpha = None, 0.3

                            plt.plot(times_, pobj_, marker=marker, label=label,
                                     alpha=alpha, color=color)
                            # reuse color for next lines
                            # color = ax.lines[-1].get_color()
                            # don't duplicate label for next lines
                            label = None

                if normalize_method not in [None, 'short']:
                    ax.set_xscale('log')
                    ax.set_yscale('log')

            plt.xlabel('Time (s)')
            if normalize_method in [None, 'short']:
                plt.ylabel('objective')
                if normalize_method == 'short' and tmax != []:
                    xmax = 10 ** np.mean(np.log10(tmax)) / 5
                    plt.xlim(-xmax / 20, xmax)
            elif normalize_method == 'diff':
                plt.ylabel('diff(objective)')
            elif normalize_method == 'last':
                plt.ylabel('(objective_i - best_i) / (init - best_i)')
            else:
                plt.ylabel('(objective - best) / best')

            # ---- Cleaner fig for the paper
            # plt.legend(loc=0, ncol=1)
            plt.ylabel('')

            # plt.ylim(ymin=ymin / 10)
            # plt.title('K = %d, L = %d' % (n_atoms, n_times_atom))
            plt.gca().tick_params(axis='x', which='both', bottom='off',
                                  top='off')
            plt.gca().tick_params(axis='y', which='both', left='off',
                                  right='off')
            plt.grid(True)
            plt.tight_layout()

            try:
                reg = min(this_res_2['reg'].unique())
            except:
                reg = None
            this_save_name = (save_name + '_bench_K%d_L%d_r%s' %
                              (n_atoms, n_times_atom, reg))
            this_save_name = this_save_name.replace(' ', '_').replace('.', '')
            this_save_name = this_save_name.replace(',', '').replace('=', '')
            this_save_name = this_save_name.replace('(', '').replace(')', '')
            fig.savefig(this_save_name + '.png', dpi=150)


def plot_barplot(all_results_df, threshold, normalize_method, save_name):
    all_results_df = all_results_df.sort_values(by=['reg'], kind='mergesort')

    if normalize_method is None:
        return None
    save_name += '_%s_%s' % (normalize_method, threshold)
    labels = all_results_df['label'].unique()
    to_plot = []
    iterator = itertools.product(
        all_results_df['n_channels'].unique(),
        all_results_df['n_atoms'].unique(),
        all_results_df['n_times_atom'].unique(),
        all_results_df['n_times'].unique(),
        all_results_df['reg'].unique(),
        labels, )
    for n_channels, n_atoms, n_times_atom, n_times, reg, label in iterator:
        setting = 'T_%d, P=%d, K=%d, L=%d' % (n_times, n_channels, n_atoms,
                                              n_times_atom)
        this_res = all_results_df
        this_res = this_res[this_res['n_atoms'] == n_atoms]
        this_res = this_res[this_res['n_times_atom'] == n_times_atom]
        this_res = this_res[this_res['n_times'] == n_times]
        this_res = this_res[this_res['n_channels'] == n_channels]
        this_res = this_res[this_res['reg'] == reg]
        this_res = this_res[this_res['label'] == label]

        if this_res.size == 0:
            continue

        # aggregate all runs (different random states)
        times = this_res['times']
        pobj = this_res['pobj']

        pobj = normalize_pobj(pobj, None, normalize_method)

        # find first time instant where pobj go below threshold
        first_time_list = []
        for pobj_, times_ in zip(pobj, times):
            idx = np.where(pobj_ < threshold)[0]
            if idx.size != 0:
                first_time_list.append(times_[idx[0]])
        first_time_list = np.array(first_time_list)

        mean = np.exp(np.mean(np.log(first_time_list)))
        std = None
        to_plot.append((reg, first_time_list, mean, std, label, setting))

    to_plot_df = pd.DataFrame(to_plot, columns=[
        'reg', 'first_time', 'mean', 'std', 'label', 'setting'
    ])

    settings = to_plot_df['setting'].unique()

    for setting in settings:
        this_to_plot_df = to_plot_df[to_plot_df['setting'] == setting]

        # # remove last point which have failed
        # if 'P=5' in setting:
        #     this_to_plot_df = this_to_plot_df[this_to_plot_df['reg'] < 50.]
        # if 'P=25' in setting:
        #     this_to_plot_df = this_to_plot_df[this_to_plot_df['reg'] < 75.]

        labels = this_to_plot_df['label'].unique()
        width = 1. / (labels.size + 2)  # the width of the bars
        colors = color_palette(len(labels))

        regs = this_to_plot_df['reg'].unique()
        regs.sort()
        x_positions = np.arange(regs.size)

        fig = plt.figure(figsize=(11, 4))
        ax = fig.gca()
        rect_list = []
        for i, label in enumerate(labels):
            this_df = this_to_plot_df[this_to_plot_df['label'] == label]

            hatch = '//' if 'Proposed' in label else ''

            rect = ax.bar(left=x_positions + i * width,
                          height=np.array(this_df['mean']), width=width,
                          label=label, hatch=hatch, color=colors[i])
            rect_list.append(rect)
            # yerr=np.array(this_df['std']),

            # color = rect[0].get_facecolor()
            for j, first_time in enumerate(this_df['first_time']):
                ax.plot(
                    np.ones_like(first_time) * x_positions[j] + i * width,
                    first_time, '_', color='k')

        offset = (labels.size - 1) / 2.0 * width
        ax.set_xticks(x_positions + offset)
        ax.set_xticklabels([r'$\lambda=%s$' % r for r in regs], ha='center')
        ax.set_yscale("log")

        plt.ylabel('Time (s)')
        plt.grid(True)
        # plt.title('Time to reach a relative precision of %s' % threshold)
        plt.tight_layout()

        # legend to the top
        plt.legend()
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        if len(labels) > 3:
            ncol = 2
            top = 0.75
        else:
            ncol = 3
            top = 0.85
        fig.legend(rect_list, labels, loc='upper center', ncol=ncol,
                   columnspacing=0.8)
        ax.legend_.remove()
        fig.subplots_adjust(top=top)

        this_save_name = '%s_%s' % (save_name, setting)
        this_save_name = this_save_name.replace(' ', '_').replace('.', '')
        this_save_name = this_save_name.replace(',', '').replace('=', '')
        this_save_name = this_save_name.replace('(', '').replace(')', '')
        fig.savefig(this_save_name + '.png')
        fig.savefig(this_save_name + '.pdf')


def change_label(data_frame, old, new):
    """Change a method label"""
    mask = data_frame['label'] == old
    data_frame.loc[mask, 'label'] = new
    return data_frame


def clean_data_frame(data_frame):
    """Various cleaning"""
    data_frame = change_label(data_frame, old='Wohlberg (2017)',
                              new='Garcia-Cardona et al (2017)')
    data_frame = change_label(data_frame, old='LGCD (1 channel)',
                              new='Proposed (univariate)')
    data_frame = change_label(data_frame, old='LGCD (full rank)',
                              new='Proposed (multivariate)')
    data_frame = change_label(data_frame, old='LGCD (rank 1)',
                              new='Proposed (multichannel)')
    data_frame = change_label(data_frame, old='Jas & al (2017) LBFGS',
                              new='Jas et al (2017) LBFGS')
    data_frame = change_label(data_frame, old='Jas & al (2017) FISTA',
                              new='Jas et al (2017) FISTA')

    data_frame = change_label(data_frame, old='Proposed (multichannel)',
                              new='Proposed (rank-1)')
    return data_frame


##############################################################################
# load the results from file
all_results_df = None
for load_name in os.listdir('figures'):
    load_name = os.path.join('figures', load_name)
    if (load_name[-4:] == '.pkl' and
            ('run' in load_name or 'debug' in load_name)):
        print("load %s" % load_name)
        data_frame = pd.read_pickle(load_name)
    else:
        continue

    data_frame = clean_data_frame(data_frame)

    if all_results_df is not None:
        all_results_df = pd.concat([all_results_df, data_frame],
                                   ignore_index=True)
    else:
        all_results_df = data_frame

    threshold = 1e-3
    normalize_method = None
    save_name = load_name[:-4]

    # plot each convergence plot
    for normalize_method in [None, 'short', 'best', 'last']:
        # plot_convergence(data_frame, threshold, normalize_method, save_name)
        plt.close('all')

threshold = 1e-3
normalize_method = 'last'
save_name = os.path.join('figures', 'all')

# plot the aggregation of all results
plot_barplot(all_results_df, threshold, normalize_method, save_name)

# plt.show()
plt.close('all')
