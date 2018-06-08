#! /usr/bin/env python
import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

descr = """Convolutional dictionary learning for noisy signals"""

DISTNAME = 'multicsc'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/multicsc/multicsc.git'
VERSION = '0.1.dev0'


def get_requirements():
    """Return the requirements of the projects in requirements.txt"""
    with open('requirements.txt') as f:
        requirements = [r.strip() for r in f.readlines()]
    return [r for r in requirements if r != '']


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
          packages=[
              'multicsc'
          ],
          install_requires=get_requirements()
          )
