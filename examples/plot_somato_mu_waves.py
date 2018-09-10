import mne
import numpy as np
import matplotlib.pyplot as plt

from alphacsc import BatchCDL
from alphacsc.datasets.somato import load_data


def plot_atom(cdl, i_atom, sfreq=150):
    """Plotting function which displays the spatial and temporal map of an atom
    """
    n_plots = 3
    figsize = (n_plots * 3.5, 5)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)

    # Plot the spatial map of the learn atom
    ax = axes[0, 0]
    u_hat = cdl.u_hat_[i_atom]
    mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
    ax.set(title='Learned spatial pattern')

    # Plot the temporal pattern of the learn atom
    ax = axes[0, 1]
    v_hat = cdl.v_hat_[i_atom]
    t = np.arange(v_hat.size) / sfreq
    ax.plot(t, v_hat)
    ax.set(xlabel='Time (sec)', title='Learned temporal waveform')
    ax.grid(True)

    # Plot the psd of the time atom
    ax = axes[0, 2]
    psd = np.abs(np.fft.rfft(v_hat)) ** 2
    frequencies = np.linspace(0, sfreq / 2.0, len(psd))
    ax.semilogy(frequencies, psd)
    ax.set(xlabel='Frequencies (Hz)', title='Power Spectral Density')
    ax.grid(True)
    ax.set_xlim(0, 30)

    plt.tight_layout()
    plt.show()


# Define the parameters
sfreq = 150.
X, info = load_data(epoch=True, sfreq=sfreq)
n_trials, n_channels, n_times = X.shape

# Define the shape of the dictionary
n_atoms = 25
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

# Set parameter for our dictionary learning algorithm
reg = .2
n_iter = 50

cdl = BatchCDL(
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    reg=reg,
    n_iter=n_iter,
    eps=2e3,
    solver_z="lgcd",
    solver_z_kwargs={'tol': 1e-1, 'max_iter': 1000},
    uv_constraint='separate',
    solver_d='alternate_adaptive',
    solver_d_kwargs={'max_iter': 300},
    D_init='chunk',
    lmbd_max="scaled",
    verbose=10,
    random_state=0,
    n_jobs=6)

# Fit the model and learn some atoms
cdl.fit(X)

# Plotting learned atom 4, which displays a mu-shape in its temporal pattern
i_atom = 4
plot_atom(cdl, i_atom, sfreq)
