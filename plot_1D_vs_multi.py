import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        'Compare dictionary retrieval capabilities with univariate and '
        'multichannel CSC for different SNR.')
    parser.add_argument('--fname', type=str, default='figures/rank1_snr.pkl',
                        help='Name of the file to plot from.')

    args = parser.parse_args()
    fname = args.fname
    all_results_df = pd.read_pickle(fname)

    fig = plt.figure(figsize=(12, 9))
    span_n_channels = all_results_df.run_n_channels.unique()
    span_sigma = all_results_df.sigma.unique()
    for n_channels in span_n_channels:
        curve = []
        results_n_channel = all_results_df[
            all_results_df['run_n_channels'] == n_channels]
        print("{} -> {}".format(n_channels,
                                results_n_channel.reg.loc[results_n_channel.score.idxmin()]))
        for sigma in span_sigma:
            results = results_n_channel[results_n_channel['sigma'] == sigma]
            curve += [results.score.min()]
        plt.loglog(span_sigma, curve, label=n_channels)

    # score_d = np.array(all_results_df.score_d)
    # score_uv = np.array(all_results_df.score_uv)
    # plt.loglog(sigma, np.sqrt(score_uv), label="multivariate")
    # plt.loglog(sigma, np.sqrt(score_d), label="univariate")
    plt.legend()
    plt.ylabel("$\\frac{\||D - \hat D||_2}{||D||_2}$", fontsize=18)
    plt.xlabel("$\sigma$", fontsize=18)
    plt.tight_layout()
    plt.savefig(fname.replace("pkl", "png"), dpi=150)
