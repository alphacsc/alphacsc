# Rank one convolutional dictionary learning based on torch and line search SGD

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from alphacsc.utils.dictionary import tukey_window
from alphacsc.init_dict import init_dictionary


class ConvSignalDataset(torch.utils.data.Dataset):
    """
    Dataset for Stochastic Deep CDL

    Parameters
    ----------
    data: np.array
        Data to be processed
    window: int
        Size of minibatches window.
    """
    def __init__(self, data, window):
        super().__init__()
        self.data = data
        self.window = window

    def __getitem__(self, idx):
        return self.data[:, idx:(idx+self.window)]

    def __len__(self):
        return self.data.shape[1] - self.window


class StoDeepCDLRank1(nn.Module):
    """
    Stochastic Deep CDL with rank one constraint

    Parameters
    ----------
    n_components: int
        Number of atoms.
    n_iter: int
        Number of unrolled iterations.
    lambd: float
        Regularization parameter in the lasso.
    kernel_size: int
        Size of the convolutional kernels in the dictionary.
    batch_window: int
        Size of minibatches window.
    device: str
        Device on which the algorithm should run. If None,
        the computations are performed on gpu if available.
    learn_steps: bool
        True if the step sizes should be learned.
    algo: str
        Which sparse coding algorithm to use ['fista', 'ista'].
    backprop: bool
        True if backpropagation should be use to compute the gradient.
        This can lead to large memory usage.
    epochs: int
        Number of epochs with fixed steps sizes.
    epochs_step_size: int
        Number of epochs to learn the steps sizes
    iterations_per_epoch: int
        Number of iterations on the data per epoch.
        If None, iterates on all the data at each epoch
    mini_batch_size: int
        Size of the minibatches
    etamax: float
        Starting maximal step size in the line search
    c: float
        Stability parameter in Armijo's line search
    beta: float
        Step size decreasing ratio in the line search
    gamma: float
        Maximal step size decreasing ratio in the line search
    """
    def __init__(self, n_components, n_iter, kernel_size, lambd=0.1,
                 batch_window=None, device=None, learn_steps=False,
                 algo="fista", backprop=False, epochs=10, epochs_step_size=0,
                 iterations_per_epoch=None, mini_batch_size=20,
                 etamax=1, c=None, beta=0.5, gamma=0.5):

        super().__init__()

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Regularization parameter
        self.lambd = lambd

        # Algorithm parameters
        self.algo = algo
        self.n_iter = n_iter
        self.backprop = backprop
        self.learn_steps = learn_steps if backprop else False

        # Number of atoms
        self.n_components = n_components

        # Convolution
        self.conv = torch.nn.functional.conv1d
        self.convt = torch.nn.functional.conv_transpose1d
        self.kernel_size = kernel_size

        # Tukey window
        self.window_tukey = torch.tensor(
            tukey_window(self.kernel_size),
            dtype=torch.float,
            device=self.device
        )[None, None, :]

        # Line search parameters
        if batch_window is None:
            self.batch_window = 10 * self.kernel_size
        else:
            self.batch_window = batch_window

        self.mini_batch_size = mini_batch_size
        self.iterations_per_epoch = iterations_per_epoch
        self.c = c
        self.etamax = etamax
        self.beta = beta
        self.gamma = gamma
        self.epochs = epochs
        self.epochs_step_size = epochs_step_size

        # Dictionary
        self.u = None
        self.v = None

    def rescale(self):
        """
        Constrains the dictionary to have normalized atoms
        """
        with torch.no_grad():
            norm_col_u = torch.norm(self.u, dim=1, keepdim=True)
            norm_col_u[torch.nonzero((norm_col_u == 0), as_tuple=False)] = 1
            self.u /= norm_col_u

            norm_col_v = torch.norm(self.v, dim=2, keepdim=True)
            norm_col_v[torch.nonzero((norm_col_v == 0), as_tuple=False)] = 1
            self.v /= norm_col_v
        return norm_col_v, norm_col_u

    def unscale(self, norm_v, norm_u):
        """
        Cancels the scaling using norms previously computed

        Parameters
        ----------
        norm_v : torch.tensor(n_atoms)
            Norms of the atoms of v before scaling
        norm_u : torch.tensor(n_atoms)
            Norms of the atoms of u before scaling
        """
        with torch.no_grad():
            self.v *= norm_v
            self.u *= norm_u

    @property
    def u_hat_(self):
        return self.u.to("cpu").detach().numpy()

    @property
    def v_hat_(self):
        with torch.no_grad():
            D = self.v * self.window_tukey
        return D.to("cpu").detach().numpy()

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT
        """
        with torch.no_grad():
            dictionary = self.u * self.v
            fourier_dico = fft.fft(dictionary, dim=2)
            self.lipschitz = torch.max(
                torch.real(fourier_dico * torch.conj(fourier_dico)),
                dim=2
            )[0].sum().item()
            if self.lipschitz == 0:
                self.lipschitz = 1

    def cost(self, y, x):
        """
        LASSO cost function

        Parameters
        ----------
        y : torch.tensor
            Data
        x : torch.tensor
            Sparse codes

        Returns
        -------
        float
            Value of the loss
        """
        dictionary = self.u * self.v
        D = dictionary * self.window_tukey
        signal = self.convt(x, D)
        res = signal - y
        l2 = (res * res).sum()
        l1 = torch.abs(x).sum()

        return 0.5 * l2 + self.lambd * l1

    def forward(self, y):
        """
        (F)ISTA-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (number of data, width, height)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape
            (number of data, n_components,
            width - kernel_size + 1)
            Approximation of the sparse code associated to y
        """
        # Initialization equal 0
        out = torch.zeros(
            (y.shape[0],
             self.n_components,
             y.shape[2] - self.kernel_size + 1),
            dtype=torch.float,
            device=self.device
        )

        if self.algo == "fista":
            out_old = out.clone()
            t_old = 1

        # Compute steps with Lipschitz constant
        steps = self.steps / self.lipschitz
        dictionary = self.u * self.v

        # Compute current dictinary after Tukey window
        D = dictionary * self.window_tukey

        for i in range(self.n_iter):
            # Gradient descent
            result1 = self.convt(out, D)
            result2 = self.conv(
                (result1 - y),
                D
            )

            out = out - steps[i] * result2
            thresh = torch.abs(out) - steps[i] * self.lambd
            out = torch.sign(out) * F.relu(thresh)

            if self.algo == "fista":
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
                z = out + ((t_old-1) / t) * (out - out_old)
                out_old = out.clone()
                t_old = t
                out = z

        return out

    def stoch_line_search(self, batch, eta, loss, state):
        """
        Stochastic line search gradient descent

        Parameters
        ----------
        batch : torch.tensor
            Data
        eta : float
            Starting step size
        loss : float
            Current value of the loss
        state : bool
            True if the steps sizes should be updated

        Returns
        -------
        float
            Next step size
        """
        ok = False
        norm_u = None
        norm_v = None
        old_eta = eta

        if not state:
            norm_grad = torch.sum(self.v.grad ** 2)\
                + torch.sum(self.u.grad ** 2)
        elif state:
            norm_grad = torch.sum(self.v.grad ** 2)\
                + torch.sum(self.u.grad ** 2)\
                + torch.sum(self.steps.grad ** 2)

        with torch.no_grad():
            # Learning step
            self.v -= self.beta * eta * self.v.grad
            self.u -= self.beta * eta * self.u.grad

            if state:
                self.steps -= self.beta * eta * self.steps.grad

            init = True

            while not ok:
                if not init:
                    # Unscaling
                    self.unscale(norm_v, norm_u)
                    # Backtracking
                    self.v -= (self.beta-1)\
                        * eta * self.v.grad
                    self.u -= (self.beta-1)\
                        * eta * self.u.grad
                    if state:
                        self.steps -= (self.beta-1) * eta * self.steps.grad
                else:
                    init = False

                # Rescaling
                norm_v, norm_u = self.rescale()

                # Computing step
                self.compute_lipschitz()

                # Computing loss with new parameters
                current_cost = self.cost(batch, self.forward(batch)).item()

                # Stopping criterion
                if current_cost < loss - self.c * eta * norm_grad:
                    ok = True
                else:
                    eta *= self.beta

                if eta < 1e-20:
                    self.v += eta * self.v.grad
                    self.u += eta * self.u.grad
                    if state:
                        self.steps += eta * self.steps.grad
                    ok = True

        return old_eta

    def train(self, epochs, state):
        """
        Training function, with stochastic line search

        Parameters
        ----------
        epochs : int
            Number of epochs
        state : bool
            True if the step sizes should be updated

        Returns
        -------
        float
            Value of the loss after training
        """

        for i in range(epochs):
            avg_loss = 0

            for idx, data in enumerate(self.dataloader):

                if self.iterations_per_epoch is not None:
                    if idx >= self.iterations_per_epoch:
                        break

                if self.device != "cpu":
                    data = data.cuda(self.device)

                data = data.float()

                # Forward pass
                if self.backprop:
                    out = self.forward(data)
                else:
                    with torch.no_grad():
                        out = self.forward(data)

                # Computing loss and gradients
                loss = self.cost(data, out)
                loss.backward()

                avg_loss = idx * avg_loss / (idx+1)\
                    + (1 / (idx+1)) * loss.item()

                # Optimizing
                if i == 0:
                    eta = self.etamax
                else:
                    eta *= self.gamma **\
                        (self.mini_batch_size / self.batch_size)

                eta = self.stoch_line_search(data, eta, loss.item(), state)

                # Putting the gradients to zero
                self.v.grad.zero_()
                self.u.grad.zero_()
                if state:
                    self.steps.grad.zero_()

            print(avg_loss)

        return loss.item()

    def fit(self, data_y):
        """
        Learn a dictionary from the data

        Parameters
        ----------
        data_y : np.array(n_channels, n_time)
            Data

        Returns
        -------
        float
            Value of the loss after training
        """
        # Dimension
        self.dim_y = data_y.shape[1]
        self.n_channels = data_y.shape[0]

        data_y_norm = data_y / data_y.std()

        init = init_dictionary(
            data_y_norm[None, :, :], self.n_components, self.kernel_size
        )

        u = init[:, :self.n_channels][:, :, None]
        v = init[:, self.n_channels:][:, None, :]

        self.u = nn.Parameter(
            torch.tensor(
                u,
                dtype=torch.float,
                device=self.device
            )
        )

        self.v = nn.Parameter(
            torch.tensor(
                v,
                dtype=torch.float,
                device=self.device
            )
        )

        # Scaling and computing step
        self.rescale()
        self.steps = nn.Parameter(
            torch.ones(self.n_iter, device=self.device, dtype=torch.float)
        )
        self.compute_lipschitz()

        # Optimization parameters
        if self.mini_batch_size is None or self.mini_batch_size > self.dim_y:
            self.mini_batch_size = self.dim_y - self.batch_window

        if self.iterations_per_epoch is not None:
            self.batch_size = self.mini_batch_size * self.iterations_per_epoch
        else:
            self.batch_size = self.dim_y

        if self.c is None:
            # Heuristic
            self.c = 10 / self.mini_batch_size

        # Dataset
        dataset = ConvSignalDataset(data_y_norm, self.batch_window)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            pin_memory=True
        )

        with torch.no_grad():
            sample = next(iter(self.dataloader)).cuda(self.device).float()
            # Compute lambda max
            self.lambd *= torch.max(
                torch.abs(self.conv(sample, self.u * self.v))
            )

        # Learning dictionary
        loss = self.train(self.epochs, state=0)

        # Learning step sizes
        if self.learn_steps:
            loss = self.train(epochs=self.epochs_step_size, state=1)

        return loss
