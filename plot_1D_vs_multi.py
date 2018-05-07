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

    sigma = np.array(all_results_df.sigma)
    score_d = np.array(all_results_df.score_d)
    score_uv = np.array(all_results_df.score_uv)

    fig = plt.figure(figsize=(12, 9))
    plt.loglog(sigma, np.sqrt(score_uv), label="multivariate")
    plt.loglog(sigma, np.sqrt(score_d), label="univariate")
    plt.legend()
    plt.ylabel("$\\frac{\||D - \hat D||_2}{||D||_2}$", fontsize=18)
    plt.xlabel("$\sigma$", fontsize=18)
    plt.tight_layout()
    plt.savefig(fname.replace("pkl", "png"), dpi=150)
