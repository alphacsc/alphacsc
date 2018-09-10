import warnings


try:
    import numba
    from numba import jit
except ImportError:  # pragma: no cover
    def jit(*args, **kwargs):
        def pass_through(f):
            return f
        return pass_through

    class dummy_numba_type:
        def __getitem__(self, key):
            return None

    numba = object()
    numba.int64 = dummy_numba_type()
    numba.float64 = dummy_numba_type()

    warnings.warn("numba is not installed on your system. The alphacsc will "
                  "be slow. Please use `pip install numba` to have a faster "
                  "computations.")
