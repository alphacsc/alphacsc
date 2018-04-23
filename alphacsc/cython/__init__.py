import pyximport
pyximport.install()
from .sparse_conv import _fast_sparse_convolve_multi
from .sparse_conv import _fast_sparse_convolve_multi_uv
from .compute_ztz import _fast_compute_ztz
