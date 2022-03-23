#! /usr/bin/env python
from setuptools import setup, Extension
import numpy as np


# Add cython extensions
kmc2 = Extension('alphacsc.other.kmc2.kmc2',
                 sources=['alphacsc/other/kmc2/kmc2.pyx'],
                 extra_compile_args=['-O3'],
                 include_dirs=[np.get_include()])
modules = [kmc2]


if __name__ == "__main__":
    from Cython.Build import cythonize

    setup(ext_modules=cythonize(modules))
