import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    load_name = "figures/rank1_snr.pkl"
    all_results_df = pd.read_pickle(load_name)

    sigma = np.array(all_results_df.sigma)
    score_d = np.array(all_results_df.score_d)
    score_uv = np.array(all_results_df.score_uv)

    fig = plt.figure(figsize=(12, 9))
    plt.semilogy(sigma, np.sqrt(score_uv), label="multivariate")
    plt.semilogy(sigma, np.sqrt(score_d), label="univariate")
    plt.legend()
    plt.ylabel("$\\frac{\||D - \hat D||_2}{||D||_2}$", fontsize=18)
    plt.xlabel("$\sigma$", fontsize=18)
    plt.tight_layout()
    plt.savefig("figures/rank1_snr.png", dpi=150)
