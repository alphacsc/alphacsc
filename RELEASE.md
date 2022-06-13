# Release procedure

This page describes the procedure to release to PyPI.

### Before release

Before each release [switcher.json](doc/_static/switcher.json) file should be
updated with the latest release version. For example, `the switcher.json` file for
latest stable release `0.3`:

```json
[
  {
    "name": "dev",
    "version": "dev",
    "url": "/dev/"
  },
  {
    "name": "0.3 (stable)",
    "version": "0.3",
    "url": "/stable/"
  }
]
```
becomes:

```json
[
  {
    "name": "dev",
    "version": "dev",
    "url": "/dev/"
  },
  {
    "name": "0.3",
    "version": "0.3",
    "url": "/0.3/"
  },
  {
    "name": "0.4.0 (stable)",
    "version": "0.4.0",
    "url": "/stable/"
  }
]
```
for release `0.4.0`.


### Automatic procedure

Create a [Github release](https://github.com/alphacsc/alphacsc/releases) with a version tag (e.g. "v1.2.3").
This will trigger a Github action which uploads the package to PyPI using the version tag as a version number.

### Manual procedure

Manually launch the Github action named "Upload to Pypi".
You can specify a version number (e.g. "1.2.3").
If no version number is specified, it is rendered from the last tag (see [versioning scheme here](https://github.com/pypa/setuptools_scm/#default-versioning-scheme])).
