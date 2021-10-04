#! /usr/bin/env python
from setuptools import setup, Extension
import numpy as np


# Add cython extensions
kmc2 = Extension('alphacsc.other.kmc2.kmc2',
                 sources=['alphacsc/other/kmc2/kmc2.pyx'],
                 extra_compile_args=['-O3'],
                 include_dirs=[np.get_include()])
sdtw = Extension('alphacsc.other.sdtw.soft_dtw_fast',
                 sources=['alphacsc/other/sdtw/soft_dtw_fast.pyx'],
                 include_dirs=[np.get_include()])
modules = [kmc2, sdtw]

# Create the alphacsc.cython modules
other_modules = [
    "compute_ztX",
    "compute_ztz",
    "coordinate_descent",
    "sparse_conv",
]
for m in other_modules:
    modules.append(
        Extension("alphacsc.cython_code.{}".format(m),
                  sources=["alphacsc/cython_code/{}.pyx".format(m)],
                  include_dirs=[np.get_include()]))

if __name__ == "__main__":
    from Cython.Build import cythonize

    setup(ext_modules=cythonize(modules))
