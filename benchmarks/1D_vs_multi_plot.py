"""Benchmark multiple channels vs a single channel for dictionary recovery.

This script plots the results saved by the script 1D_vs_multi_run.py, which
should be run beforehand.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        'Compare dictionary retrieval capabilities with univariate and '
        'multichannel CSC for different SNR.')
    parser.add_argument('--file-name', type=str,
                        default='figures/rank1_snr.pkl',
                        help='Name of the result file to plot from.')
    parser.add_argument('--pdf', action='store_true',
                        help='Output pdf figures for final version.')

    args = parser.parse_args()
    file_name = args.file_name
    if not os.path.exists(file_name):
        raise FileNotFoundError("Could not find result file '{}'. Make sure "
                                "to run 1D_vs_multi_run.py before using this "
                                "script.")

    extension = "pdf" if args.pdf else "png"

    # Load the results
    all_results_df = pd.read_pickle(file_name)

    # Setup the figure
    fontsize = 14
    figsize = (6, 3.4)
    mpl.rc('mathtext', fontset='cm')
    fig = plt.figure(figsize=figsize)
    normalize = mcolors.LogNorm(vmin=1, vmax=50)
    colormap = plt.cm.get_cmap('viridis')

    ############################################################
    # Plot recovery score against the noise level for different
    # channel numbers
    ############################################################

    span_n_channels = all_results_df.run_n_channels.unique()
    span_sigma = all_results_df.sigma.unique()
    for n_channels in span_n_channels:
        curve = []
        results_n_channel = all_results_df[
            all_results_df['run_n_channels'] == n_channels]
        for sigma in span_sigma:
            results = results_n_channel[results_n_channel['sigma'] == sigma]
            results = results.groupby(['random_state']).min()
            curve += [results.score.mean()]
        color = colormap(normalize(n_channels))
        plt.loglog(span_sigma, curve, color=color,
                   label="$P={}$".format(n_channels))

    plt.legend(loc=2, fontsize=fontsize)
    plt.ylabel(r"score($\widehat v$)", fontsize=fontsize)
    plt.xlabel(r"Noise level $\eta$", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(file_name.replace("pkl", extension), dpi=150)

    ##############################################################
    # For each channel number, plot the recovered atoms compared
    # to the initial ones.
    ##############################################################

    sig = all_results_df.sigma.unique()[5]
    print("eta = {:.2e}".format(sig))
    for P in span_n_channels:
        if P == 1:
            continue
        plt.figure(figsize=figsize)
        res_sig = all_results_df[all_results_df.sigma == sig]
        lines = []
        for n_channels, color in [(1, 'C0'), (P, 'C1')]:
            res = res_sig[res_sig.run_n_channels == n_channels]
            i0 = res.score.idxmin()
            uv_hat = res.uv_hat[i0]
            print("[P={}] Best lmbd {}".format(n_channels, res.reg[i0]))
            uv = res.uv[i0]
            s = np.dot(uv[:, -64:], uv_hat[:, -64:].T)
            if np.trace(abs(s)) >= np.trace(abs(s)[::-1]):
                uv_hat *= np.sign(np.diag(s))[:, None]
            else:
                uv_hat *= np.sign(np.diag(s[::-1]))[:, None]

            ll = plt.plot(uv_hat[:, -64:].T, color=color, label=n_channels)
            lines += [ll[0]]
        ll = plt.plot(uv[:, -64:].T, "k--", label="GT")
        lines += [ll[0]]
        plt.legend(lines, ['$P=1$', '$P={}$'.format(P), "GT"], loc=8,
                   fontsize=fontsize, ncol=3, columnspacing=.5)
        plt.xlabel("Times", fontsize=fontsize)
        plt.ylabel("Atoms", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(file_name.replace(".pkl", "_uv_hat_P{}.{}").format(
            P, extension), dpi=150)
