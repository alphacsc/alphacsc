import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


fontsize = 14
figsize = (6, 3.4)
mpl.rc('mathtext', fontset='cm')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        'Compare dictionary retrieval capabilities with univariate and '
        'multichannel CSC for different SNR.')
    parser.add_argument('--fname', type=str, default='figures/rank1_snr.pkl',
                        help='Name of the file to plot from.')
    parser.add_argument('--bck', action='store_true',
                        help='Generate back-up figure.')

    args = parser.parse_args()
    fname = args.fname
    if args.bck:
        fname = "figures/1D_vs_multi.pkl"
    all_results_df = pd.read_pickle(fname)

    normalize = mcolors.LogNorm(vmin=1, vmax=50)
    colormap = plt.cm.get_cmap('viridis')

    fig = plt.figure(figsize=figsize)
    span_n_channels = all_results_df.run_n_channels.unique()
    span_sigma = all_results_df.sigma.unique()
    for n_channels in span_n_channels:
        if args.bck and n_channels not in [1, 5, 25, 50]:
            continue
        curve = []
        results_n_channel = all_results_df[
            all_results_df['run_n_channels'] == n_channels]
        for sigma in span_sigma:
            results = results_n_channel[results_n_channel['sigma'] == sigma]
            if args.bck:
                curve += [results.score.min()]
            else:
                results = results.groupby(['random_state']).min()
                curve += [results.score.mean()]
        color = colormap(normalize(n_channels))
        plt.loglog(span_sigma, curve, color=color,
                   label="$P={}$".format(n_channels))

    # # Colorbar setup
    # s_map = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
    # s_map.set_array(span_n_channels)

    # # If color parameters is a linspace, we can set boundaries in this way
    # boundaries = np.linspace(.5, 50.5, 51)

    # # Use this to show a continuous colorbar
    # cbar = fig.colorbar(s_map, spacing='proportional', ticks=span_n_channels,
    #                     format='%2i')

    # cbarlabel = r'# of channels $P$'
    # cbar.set_label(cbarlabel, fontsize=20)

    plt.legend(loc=2, fontsize=fontsize)
    plt.ylabel("score($\widehat v$)", fontsize=fontsize)
    plt.xlabel("Noise level $\eta$", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(fname.replace("pkl", "png"), dpi=150)

    sig = all_results_df.sigma.unique()[12]
    print("eta = {:.2e}".format(sig))
    for P in span_n_channels:
        if P == 1:
            continue
        plt.figure(figsize=figsize)
        res_sig = all_results_df[all_results_df.sigma == sig]
        lines = []
        for n_chan, color in [(1, 'C0'), (P, 'C1')]:
            res = res_sig[res_sig.run_n_channels == n_chan]
            i0 = res.score.idxmin()
            uv_hat = res.uv_hat[i0]
            uv = res.uv[i0]
            s = np.dot(uv[:, -64:], uv_hat[:, -64:].T)
            if np.trace(abs(s)) >= np.trace(abs(s)[::-1]):
                uv_hat *= np.sign(np.diag(s))[:, None]
            else:
                uv_hat *= np.sign(np.diag(s[::-1]))[:, None]

            ll = plt.plot(uv_hat[:, -64:].T, color=color, label=n_chan)
            lines += [ll[0]]
        ll = plt.plot(uv[:, -64:].T, "k--", label="GT")
        lines += [ll[0]]
        plt.legend(lines, ['$P=1$', '$P={}$'.format(P), "GT"], loc=8,
                   fontsize=fontsize, ncol=3, columnspacing=.5)
        plt.xlabel("Times", fontsize=fontsize)
        plt.ylabel("Atoms", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(fname.replace(".pkl", "_uv_hat_P{}.png").format(P),
                    dpi=150)
