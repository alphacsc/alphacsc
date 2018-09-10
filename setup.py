#! /usr/bin/env python
import numpy as np
from setuptools import setup, Extension, find_packages


descr = """Convolutional dictionary learning for noisy signals"""

DISTNAME = 'alphacsc'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/alphacsc/alphacsc.git'
VERSION = '0.1.dev0'


def get_requirements():
    """Return the requirements of the projects in requirements.txt"""
    with open('requirements.txt') as f:
        requirements = [r.strip() for r in f.readlines()]
    return [r for r in requirements if r != '']


try:
    from Cython.Build import cythonize
    kmc2 = Extension('alphacsc.other.kmc2.kmc2',
                     sources=['alphacsc/other/kmc2/kmc2.pyx'],
                     extra_compile_args=['-O3'])
    sdtw = Extension('alphacsc.other.sdtw.soft_dtw_fast',
                     sources=['alphacsc/other/sdtw/soft_dtw_fast.pyx'])
    modules = [kmc2, sdtw]

    # Create the alphacsc.cython module
    for m in ["compute_ztX", "compute_ztz", "coordinate_descent",
              "sparse_conv"]:
        modules.append(Extension(
            "alphacsc.cython_code.{}".format(m),
            sources=["alphacsc/cython_code/{}.pyx".format(m)]))

    ext_modules = cythonize(modules)
except ImportError:
    import warnings
    warnings.warn("the optional dependency `cython` is unavailable on this "
                  "system so some functionality (D_init='kmeans') might not "
                  "work properly.")
    ext_modules = None


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
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
          install_requires=get_requirements(),
          include_dirs=[np.get_include()]
          )
