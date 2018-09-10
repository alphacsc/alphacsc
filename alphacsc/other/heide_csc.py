"""
- This code implements a solver for the paper
  "Fast and Flexible Convolutional Sparse Coding" on signal data.
- The goal of this solver is to find the common filters, the codes for each
  signal series in the dataset
and a reconstruction for the dataset.
- The common filters (or kernels) and the codes for each image is denoted as
  d and z respectively
- We denote the step to solve the filter as d-step and the codes z-step
"""
import time
import numpy as np
from scipy import linalg, signal
from scipy.fftpack import fft, ifft
from mne.utils import check_random_state

real_type = 'float64'
imaginary_type = 'complex128'


def update_d(z_hat, d_hat, size_z, size_x, rho, d, v_D,
             d_D, lambdas, gammas_D, Mtb, u_D, M, size_k_full,
             psf_radius, xi_D, xi_D_hat, verbose, max_it_d):
    """D-STEP"""
    # Precompute what is necessary for later
    [zhat_mat, zhat_inv_mat] = precompute_D_step(z_hat, size_z, rho, verbose)

    for i_d in range(max_it_d):

        v_D[0] = np.real(
            ifft(np.einsum('ijk,jk->ik', z_hat, d_hat)))
        v_D[1] = d

        # Compute proximal updates
        u = v_D[0] - d_D[0]
        theta = lambdas[0] / gammas_D[0]
        u_D[0] = (Mtb + 1.0 / theta * u) / (M + 1.0 / theta * np.ones(size_x))

        u = v_D[1] - d_D[1]
        u_D[1] = KernelConstraintProj(u, size_k_full, psf_radius)

        # Update Langrange multipliers
        d_D[0] += u_D[0] - v_D[0]
        d_D[1] += u_D[1] - v_D[1]

        # Compute new xi=u+d and transform to fft
        xi_D[0] = u_D[0] + d_D[0]
        xi_D[1] = u_D[1] + d_D[1]
        xi_D_hat[0] = fft(xi_D[0])
        xi_D_hat[1] = fft(xi_D[1])

        # Solve convolutional inverse
        d_hat = solve_conv_term_D(
            zhat_mat, zhat_inv_mat, xi_D_hat, rho, size_z)
        d = np.real(ifft(d_hat))

    return d, d_hat


def update_z(z, z_hat, d_hat, u_Z, v_Z, d_Z, lambdas,
             gammas_Z, Mtb, M, size_x, size_z, xi_Z, xi_z_hat, b,
             lambda_prior, lambda_residual, psf_radius, verbose, max_it_z):
    """Z-STEP"""
    # Precompute what is necessary for later
    dhat_flat, dhatTdhat_flat = precompute_Z_step(d_hat, size_x, verbose)
    dhatT_flat = np.ma.conjugate(dhat_flat.T)

    for i_z in range(max_it_z):

        # Compute v = [Dz,z]
        v_Z[0] = np.real(
            ifft(np.einsum('ijk,jk->ik', z_hat, d_hat)))
        v_Z[1] = z

        # Compute proximal updates
        u = v_Z[0] - d_Z[0]
        theta = lambdas[0] / gammas_Z[0]
        u_Z[0] = (Mtb + 1.0 / theta * u) / (M + 1.0 / theta * np.ones(size_x))

        u = v_Z[1] - d_Z[1]
        theta = lambdas[1] / gammas_Z[1] * np.ones(u.shape)
        # u_Z[1] = np.multiply(np.maximum(
        #     0, 1 - np.divide(theta, np.abs(u))), u)
        u_Z[1] = np.maximum(u - theta, 0)  # positivity constraint

        # Update Lagrange multipliers
        d_Z[0] += u_Z[0] - v_Z[0]
        d_Z[1] += u_Z[1] - v_Z[1]

        # Compute new xi=u+d and transform to fft
        xi_Z[0] = u_Z[0] + d_Z[0]
        xi_Z[1] = u_Z[1] + d_Z[1]

        xi_z_hat[0] = fft(xi_Z[0])
        xi_z_hat[1] = fft(xi_Z[1])

        # Solve convolutional inverse
        z_hat = solve_conv_term_Z(
            dhatT_flat, dhatTdhat_flat, xi_z_hat, gammas_Z, size_z)
        z = np.real(ifft(z_hat))

    return z, z_hat


def learn_conv_sparse_coder(b, size_kernel, max_it, tol,
                            lambda_prior=1.0, lambda_residual=1.0,
                            random_state=None, ds_init=None,
                            feasible_evaluation=True,
                            stopping_pobj=None, verbose=1,
                            max_it_d=10, max_it_z=10):
    """
    Main function to solve the convolutional sparse coding.

    Parameters
    ----------
    - b               : the signal dataset with size (num_signals, length)
    - size_kernel     : the size of each kernel (num_kernels, length)
    - max_it          : the maximum iterations of the outer loop
    - tol             : the minimal difference in filters and codes after each
                        iteration to continue

    Important variables used in the code:
    - u_D, u_Z        : pair of proximal values for d-step and z-step
    - d_D, d_Z        : pair of Lagrange multipliers in the ADMM algo for
                        d-step and z-step
    - v_D, v_Z        : pair of initial value pairs (Zd, d) for d-step and
                        (Dz, z) for z-step
    """
    rng = check_random_state(random_state)

    k = size_kernel[0]
    n = b.shape[0]

    psf_radius = int(np.floor(size_kernel[1] / 2))

    size_x = [n, b.shape[1] + 2 * psf_radius]
    size_z = [n, k, size_x[1]]
    size_k_full = [k, size_x[1]]

    # M is MtM, Mtb is Mtx, the matrix M is zero-padded in 2*psf_radius rows
    # and cols
    M = np.pad(np.ones(b.shape, dtype=real_type),
               ((0, 0), (psf_radius, psf_radius)),
               mode='constant', constant_values=0)
    Mtb = np.pad(b, ((0, 0), (psf_radius, psf_radius)),
                 mode='constant', constant_values=0)

    """Penalty parameters, including the calculation of augmented
       Lagrange multipliers"""
    lambdas = [lambda_residual, lambda_prior]
    gamma_heuristic = 60 * lambda_prior * 1 / np.amax(b)
    gammas_D = [gamma_heuristic / 5000, gamma_heuristic]
    gammas_Z = [gamma_heuristic / 500, gamma_heuristic]
    rho = gammas_D[1] / gammas_D[0]

    """Initialize variables for the d-step"""
    varsize_D = [size_x, size_k_full]
    xi_D = [np.zeros(varsize_D[0], dtype=real_type),
            np.zeros(varsize_D[1], dtype=real_type)]

    xi_D_hat = [np.zeros(varsize_D[0], dtype=imaginary_type),
                np.zeros(varsize_D[1], dtype=imaginary_type)]

    u_D = [np.zeros(varsize_D[0], dtype=real_type),
           np.zeros(varsize_D[1], dtype=real_type)]

    # Lagrange multipliers
    d_D = [np.zeros(varsize_D[0], dtype=real_type),
           np.zeros(varsize_D[1], dtype=real_type)]

    v_D = [np.zeros(varsize_D[0], dtype=real_type),
           np.zeros(varsize_D[1], dtype=real_type)]

    # d = rng.normal(size=size_kernel)
    if ds_init is None:
        d = rng.randn(*size_kernel)
    else:
        d = ds_init.copy()
    d_norm = np.linalg.norm(d, axis=1)
    d /= d_norm[:, None]

    # Initial the filters and its fft after being rolled to fit the frequency
    d = np.pad(d, ((0, 0),
                   (0, size_x[1] - size_kernel[1])),
               mode='constant', constant_values=0)
    d = np.roll(d, -int(psf_radius), axis=1)
    d_hat = fft(d)

    # Initialize variables for the z-step
    varsize_Z = [size_x, size_z]
    xi_Z = [np.zeros(varsize_Z[0], dtype=real_type),
            np.zeros(varsize_Z[1], dtype=real_type)]

    xi_z_hat = [np.zeros(varsize_Z[0], dtype=imaginary_type),
                np.zeros(varsize_Z[1], dtype=imaginary_type)]

    u_Z = [np.zeros(varsize_Z[0], dtype=real_type),
           np.zeros(varsize_Z[1], dtype=real_type)]

    # Lagrange multipliers
    d_Z = [np.zeros(varsize_Z[0], dtype=real_type),
           np.zeros(varsize_Z[1], dtype=real_type)]

    v_Z = [np.zeros(varsize_Z[0], dtype=real_type),
           np.zeros(varsize_Z[1], dtype=real_type)]

    # Initial the codes and its fft
    # z = rng.normal(size=size_z)
    z = np.zeros(size_z)
    z_hat = fft(z)

    """Initial objective function (usually very large)"""
    # obj_val = obj_func(z_hat, d_hat, b,
    #                    lambda_residual, lambda_prior,
    #                    psf_radius, size_z, size_x)
    obj_val = obj_func_2(z, d, b, lambda_prior, psf_radius,
                         feasible_evaluation)

    if verbose > 0:
        print('Init, Obj %3.3f' % (obj_val, ))

    times = list()
    times.append(0.)
    list_obj_val = list()
    list_obj_val.append(obj_val)
    """Start the main algorithm"""
    for i in range(max_it):

        start = time.time()
        z, z_hat = update_z(
            z, z_hat, d_hat, u_Z, v_Z, d_Z, lambdas,
            gammas_Z, Mtb, M, size_x, size_z, xi_Z, xi_z_hat, b,
            lambda_prior, lambda_residual, psf_radius, verbose, max_it_z)
        times.append(time.time() - start)

        # obj_val = obj_func(z_hat, d_hat, b,
        #                    lambda_residual, lambda_prior,
        #                    psf_radius, size_z, size_x)
        obj_val = obj_func_2(z, d, b, lambda_prior, psf_radius,
                             feasible_evaluation)

        if verbose > 0:
            print('Iter Z %d/%d, Obj %3.3f' % (i, max_it, obj_val))

        start = time.time()
        d, d_hat = update_d(
            z_hat, d_hat, size_z, size_x, rho, d,
            v_D, d_D, lambdas, gammas_D, Mtb, u_D, M, size_k_full,
            psf_radius, xi_D, xi_D_hat, verbose, max_it_d)
        times.append(time.time() - start)

        # obj_val = obj_func(z_hat, d_hat, b,
        #                    lambda_residual, lambda_prior,
        #                    psf_radius, size_z, size_x)
        obj_val = obj_func_2(z, d, b, lambda_prior, psf_radius,
                             feasible_evaluation)

        if verbose > 0:
            print('Iter D %d/%d, Obj %3.3f' % (i, max_it, obj_val))

        list_obj_val.append(obj_val)

        # Debug progress
        # z_comp = z

        # Termination
        # if (linalg.norm(z_diff) / linalg.norm(z_comp) < tol and
        #         linalg.norm(d_diff) / linalg.norm(d_comp) < tol):
        #     break
        if stopping_pobj is not None and obj_val < stopping_pobj:
            break

    """Final estimate"""
    z_res = z

    d_res = d
    d_res = np.roll(d_res, psf_radius, axis=1)
    d_res = d_res[:, 0:psf_radius * 2 + 1]

    Dz = np.real(ifft(np.einsum('ijk,jk->ik', z_hat, d_hat)))

    # obj_val = obj_func(z_hat, d_hat, b,
    #                    lambda_residual, lambda_prior,
    #                    psf_radius, size_z, size_x)
    # if verbose > 0:
    #     print('Final objective function %f' % obj_val)
    #
    # reconstr_err = reconstruction_err(z_hat, d_hat, b, psf_radius, size_x)
    # if verbose > 0:
    #     print('Final reconstruction error %f' % reconstr_err)

    return d_res, z_res, Dz, np.array(list_obj_val), times


def KernelConstraintProj(u, size_k_full, psf_radius):
    """Computes the proximal operator for kernel by projection"""

    # Get support
    u_proj = u
    u_proj = np.roll(u_proj, psf_radius, axis=1)
    u_proj = u_proj[:, 0:psf_radius * 2 + 1]

    # Normalize
    u_sum = np.sum(np.power(u_proj, 2), axis=1)
    u_norm = np.tile(u_sum, [u_proj.shape[1], 1]).transpose(1, 0)
    u_proj[u_norm >= 1] = u_proj[u_norm >= 1] / np.sqrt(u_norm[u_norm >= 1])

    # Now shift back and pad again
    u_proj = np.pad(u_proj, ((0, 0),
                             (0, size_k_full[1] - (2 * psf_radius + 1))),
                    mode='constant', constant_values=0)

    u_proj = np.roll(u_proj, -psf_radius, axis=1)

    return u_proj


def precompute_D_step(z_hat, size_z, rho, verbose):
    """Computes to cache the values of Z^.T and (Z^.T*Z^ + rho*I)^-1 as
       in algorithm"""

    n = size_z[0]
    k = size_z[1]

    zhat_mat = np.transpose(z_hat, [2, 0, 1])
    zhat_inv_mat = np.zeros((zhat_mat.shape[0], k, k), dtype=imaginary_type)
    inv_rho_z_hat_z_hat_t = np.zeros(
        (zhat_mat.shape[0], n, n), dtype=imaginary_type)
    z_hat_mat_t = np.transpose(np.ma.conjugate(zhat_mat), [0, 2, 1])

    # Compute z_hat * z_hat^T for each pixel
    z_hat_z_hat_t = np.einsum('knm,kmj->knj', zhat_mat, z_hat_mat_t)

    for i in range(zhat_mat.shape[0]):
        z_hat_z_hat_t_plus_rho = z_hat_z_hat_t[i]
        z_hat_z_hat_t_plus_rho.flat[::n + 1] += rho
        inv_rho_z_hat_z_hat_t[i] = linalg.pinv(z_hat_z_hat_t_plus_rho)

    zhat_inv_mat = 1.0 / rho * (np.eye(k) -
                                np.einsum('knm,kmj->knj',
                                          np.einsum('knm,kmj->knj',
                                                    z_hat_mat_t,
                                                    inv_rho_z_hat_z_hat_t),
                                          zhat_mat))

    # if verbose > 0:
    #     print('Done precomputing for D')
    return zhat_mat, zhat_inv_mat


def precompute_Z_step(dhat, size_x, verbose):
    """Computes to cache the values of D^.T and D^.T*D^ as in algorithm"""

    dhat_flat = dhat.T
    dhatTdhat_flat = np.sum(np.multiply(
        np.ma.conjugate(dhat_flat), dhat_flat), axis=1)
    # if verbose > 0:
    #     print('Done precomputing for Z')
    return dhat_flat, dhatTdhat_flat


def solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_hat, rho, size_z):
    """Solves argmin(||Zd - x1||_2^2 + rho * ||d - x2||_2^2"""

    k = size_z[1]
    sx = size_z[2]

    # Reshape to array per frequency
    xi_hat_0_flat = np.expand_dims(xi_hat[0].T, axis=2)
    xi_hat_1_flat = np.expand_dims(xi_hat[1].T, axis=2)

    x = np.zeros((zhat_mat.shape[0], k), dtype=imaginary_type)
    z_hat_mat_t = np.ma.conjugate(zhat_mat.transpose(0, 2, 1))
    x = np.einsum("ijk, ikl -> ijl", zhat_inv_mat,
                  np.einsum("ijk, ikl -> ijl",
                            z_hat_mat_t,
                            xi_hat_0_flat) +
                  rho * xi_hat_1_flat) \
        .reshape(sx, k)

    # Reshape to get back the new D^
    d_hat = x.T

    return d_hat


def solve_conv_term_Z(dhatT, dhatTdhat, xi_hat, gammas, size_z):
    """Solves argmin(||Dz - x1||_2^2 + rho * ||z - x2||_2^2"""

    sx = size_z[2]
    rho = gammas[1] / gammas[0]

    # Compute b
    xi_hat_0_rep = np.expand_dims(xi_hat[0], axis=1)
    xi_hat_1_rep = xi_hat[1]

    b = (dhatT * xi_hat_0_rep) + rho * xi_hat_1_rep

    scInverse = np.ones((1, sx)) / (rho * np.ones((1, sx)) + dhatTdhat.T)

    dhatT_dot_b = np.ma.conjugate(dhatT) * b
    dhatTb_rep = np.expand_dims(np.sum(dhatT_dot_b, axis=1), axis=1)
    x = 1.0 / rho * (b - (scInverse * dhatT) * dhatTb_rep)

    z_hat = x
    return z_hat


def obj_func(z_hat, d_hat, b, lambda_residual, lambda_prior,
             psf_radius, size_z, size_x):
    """Computes the objective function including the data-fitting and
       regularization terms"""
    # Data-fitting term
    f_z = reconstruction_err(z_hat, d_hat, b, psf_radius, size_x)

    # Regularizer
    z = ifft(z_hat)
    g_z = (lambda_prior / lambda_residual) * np.sum(z)

    f_val = f_z + g_z
    return np.real(f_val)


def reconstruction_err(z_hat, d_hat, b, psf_radius, size_x):
    """Computes the reconstruction error from the data-fitting term"""

    Dz = np.real(ifft(np.einsum('ijk,jk->ik', z_hat, d_hat)))
    err = 0.5 * linalg.norm(Dz[:, psf_radius:-psf_radius] - b) ** 2

    return err


def obj_func_2(z, d, b, lambda_prior, psf_radius, feasible_evaluation=True):
    """Alternative objective function in time domain"""
    d = d.copy()
    z = z.copy()

    z = z[:, :, 2 * psf_radius:-2 * psf_radius]
    z = z.swapaxes(0, 1)

    # get support of d in the time domain
    d = np.roll(d, psf_radius, axis=1)
    d = d[:, 0:psf_radius * 2 + 1]

    if feasible_evaluation:
        # project to unit norm
        d_norm = np.linalg.norm(d, axis=1)
        mask = d_norm >= 1
        d[mask] /= d_norm[mask][:, None]
        # update z in the opposite way
        z[mask] *= d_norm[mask][:, None, None]

    # construct X and compute the objective function with the regularization
    X = construct_X(z, d)
    obj = objective(b, X, z, reg=lambda_prior)
    return obj


def construct_X(Z, ds):
    """
    Parameters
    ----------
    z : array, shape (n_atoms, n_trials, n_times)
        The activations
    ds : array, shape (n_atoms, n_times_atom)
        The atom.

    Returns
    -------
    X : array, shape (n_trials, n_times + n_times_atom - 1)
    """
    assert Z.shape[0] == ds.shape[0]
    n_trials = Z.shape[1]
    n_times_atom = ds.shape[1]
    n_times = Z.shape[2] + n_times_atom - 1
    X = np.zeros((n_trials, n_times))
    for i in range(Z.shape[1]):
        X[i] = sum([signal.convolve(Z[k, i], d)
                    for k, d in enumerate(ds)], 0)
    return X


def objective(X, X_hat, z_hat, reg):
    obj = 0.5 * linalg.norm(X - X_hat, 'fro')**2 + reg * z_hat.sum()
    return obj
