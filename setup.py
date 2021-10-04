#! /usr/bin/env python
import os
import re
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

descr = """Convolutional dictionary learning for noisy signals"""

DISTNAME = 'alphacsc'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/alphacsc/alphacsc.git'


# Function to parse __version__ in `alphacsc`
def find_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'alphacsc', '__init__.py'), 'r') as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version()

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
