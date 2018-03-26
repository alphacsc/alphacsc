import warnings


try:
    from numba import jit
except ImportError:
    def jit():
        def pass_through(f):
            return f
        return pass_through

    warnings.warn("numba is not installed on your system. The alaphacsc will "
                  "be slow. Please use `pip install numba` to have a faster "
                  "computations.")