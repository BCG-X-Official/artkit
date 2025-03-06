.. _contributor-guide-builds:

Package Builds
==============

The build process for the PyPI and conda distributions uses the following key
files:

- ``make.py``: generic Python script for package builds. Most configuration is imported
  from `make.py <https://github.com/bcgx-pi-PID-XGF-08/llmcheck/blob/develop/make.py>`__
  which is a build script that wraps the package build, as well as exposing the matrix
  dependency definitions specified in the ``pyproject.toml`` as environment variables
- ``pyproject.toml``: metadata for PyPI, build settings and package dependencies
- ``tox.ini``: contains configurations for tox, testenv, flake8, isort, coverage report,
  and pytest
- ``condabuild/meta.yml``: metadata for conda, build settings and package dependencies


Versioning
----------

``artkit`` version numbering follows the `semantic versioning <https://semver.org/>`_
approach, with the pattern ``MAJOR.MINOR.PATCH``.
The version can be bumped in the ``src/__init__.py`` by updating the
``__version__`` string accordingly.


PyPI
----

PyPI project metadata, build settings and package dependencies
are obtained from ``pyproject.toml``. To build and then publish the package to PyPI,
use the following commands:

.. code-block:: sh

    python make.py artkit tox default
    flit publish

Please note the following:

*   Because the PyPI package index is immutable, it is recommended to do a test
    upload to `PyPI test <https://test.pypi.org/>`__ first. Ensure all metadata presents
    correctly before proceeding to proper publishing. The command to publish to test is

    .. code-block:: sh

        flit publish --repository testpypi

    which requires the specification of testpypi in a special ``.pypirc`` file
    with specifications as demonstrated `here
    <https://flit.readthedocs.io/en/latest/upload.html>`__.
*   The ``pyproject.toml`` does not provide specification for a short description
    (displayed in the top gray band on the PyPI page for the package). This description
    comes from the ``src/__init__.py`` script.
*   `flit <https://flit.readthedocs.io/en/latest/>`__ which is used here to publish to
    PyPI, also has the flexibility to support package building (wheel/sdist) via
    ``flit build`` and installing the package by copy or symlink via ``flit install``.
*   Build output will be stored in the ``dist/`` directory.


Conda
-----

conda build metadata, build settings and package dependencies
are obtained from ``meta.yml``. To build and then publish the package to conda,
use the following commands:

.. code-block:: sh

    python make.py artkit conda default
    anaconda upload --user BCG_Gamma dist/conda/noarch/<*package.tar.gz*>

Please note the following:

- Build output will be stored in the ``dist/`` directory.
- Some useful references for conda builds:

  - `Conda build tutorial
    <https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/building-conda-packages.html>`_
  - `Conda build metadata reference
    <https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html>`_

Azure DevOps CI/CD
------------------

This project uses `Azure DevOps <https://dev.azure.com/>`_ for CI/CD pipelines.
The pipelines are defined in the ``azure-pipelines.yml`` file and are divided into
the following stages:

* **code_quality_checks**: perform code quality checks for isort, black and flake8.
* **detect_build_config_changes**: detect whether the build configuration as specified
  in the ``pyproject.yml`` has been modified. If it has, then a build test is run.
* **Unit tests**: runs all unit tests and then publishes test results and coverage.
* **conda_tox_build**: build the PyPI and conda distribution artifacts.
* **Release**: see release process below for more detail.
* **Docs**: build and publish documentation to GitHub Pages.
