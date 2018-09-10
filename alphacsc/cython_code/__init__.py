
try:
    from .sparse_conv import _fast_sparse_convolve_multi
    from .sparse_conv import _fast_sparse_convolve_multi_uv
    from .compute_ztz import _fast_compute_ztz
    from .compute_ztz import _fast_compute_ztz_lil, _fast_compute_ztz_csr
    from .compute_ztX import _fast_compute_ztX
    from .coordinate_descent import subtract_zhat_to_beta, update_dz_opt
    _CYTHON_AVAILABLE = True

    __all__ = ["_fast_sparse_convolve_multi", "_fast_sparse_convolve_multi_uv",
               "_fast_compute_ztz", "_fast_compute_ztz_lil",
               "_fast_compute_ztz_csr", "_fast_compute_ztX",
               "subtract_zhat_to_beta", "update_dz_opt"]

except ImportError:  # pragma: no cover
    _CYTHON_AVAILABLE = False


def _assert_cython():
    if not _CYTHON_AVAILABLE:  # pragma: no cover
        raise NotImplementedError("cython is a required dependency for this "
                                  "part of the code.")
