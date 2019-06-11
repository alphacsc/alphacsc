#! /usr/bin/env python
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
VERSION = '0.4.dev0'

# Add cython extensions
kmc2 = Extension('alphacsc.other.kmc2.kmc2',
                 sources=['alphacsc/other/kmc2/kmc2.pyx'],
                 extra_compile_args=['-O3'])
sdtw = Extension('alphacsc.other.sdtw.soft_dtw_fast',
                 sources=['alphacsc/other/sdtw/soft_dtw_fast.pyx'])
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
                  sources=["alphacsc/cython_code/{}.pyx".format(m)]))
ext_modules = cythonize(modules)

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=open('README.rst').read(),
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        platforms='any',
        ext_modules=ext_modules,
        packages=find_packages(exclude=["tests"]),
        setup_requires=['Cython', 'numpy'],
        install_requires=[
            'mne',
            'numba',
            'numpy',
            'scipy',
            'joblib',
            'matplotlib',
            'scikit-learn',
        ],
        include_dirs=[np.get_include()], )
