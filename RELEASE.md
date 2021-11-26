# Release procedure

This page describes the procedure to release to PyPI.

### Automatic procedure

Create a [Github release](https://github.com/alphacsc/alphacsc/releases) with a version tag (e.g. "v1.2.3").
This will trigger a Github action which uploads the package to PyPI using the version tag as a version number.

### Manual procedure

Manually launch the Github action named "Upload to Pypi".
You can specify a version number (e.g. "1.2.3").
If no version number is specified, it is rendered from the last tag (see [versioning scheme here](https://github.com/pypa/setuptools_scm/#default-versioning-scheme])).
