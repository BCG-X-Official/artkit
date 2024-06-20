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


Release process
---------------

Before initiating the release process, please ensure the version number
in ``src/__init__.py`` is correct and the format conforms to semantic
versioning. If the version needs to be corrected/bumped then open a PR for the
change and merge into develop before going any further.

The release process has the following key steps:

- Create a new release branch from the tag of the latest release named
  ``release/<version>`` where ``<version>`` is the version number of the new release
- Create a new branch from the baseline branch (e.g., ``2.0.x``) named
  ``dev/<version>`` where ``<version>`` is the version number of the new release
- Opening a PR to merge ``dev/<version>`` onto ``release/<version>``.
  This will automatically run all conda/pip build tests via
  Azure Pipelines prior to allowing to merge the PR.
  This will trigger automatic upload of artifacts (conda and pip
  packages) from Azure DevOps. At this stage, it is recommended that the pip package
  build is checked using `PyPI test <https://test.pypi.org/>`__ to ensure all
  metadata presents correctly. This is important as package versions in
  PyPI proper are immutable.
- If everything passes and looks okay, merge the PR using a *merge commit*
  (not squashing).
  This will trigger the release pipeline which will:

  * Tag the release commit with version number as specified in ``src/__init__.py``
  * Create a release on GitHub for the new version, please check the `documentation
    <https://docs.github.com/en/free-pro-team@latest/github/administering-a-repository/releasing-projects-on-github>`__
    for details
  * Pre-fill the GitHub release title and description, including the changelog based on
    commits since the last release. Please note this can be manually edited to be more
    succinct afterwards
  * Attach build artifacts (conda and pip packages) to GitHub release
  * Upload build artifacts to conda/PyPI using ``anaconda upload`` and
    ``flit publish``, respectively

-  Remove any test versions for pip from PyPI test
-  Merge ``release/<version>`` back onto the baseline branch from which
   ``dev/<version>`` was branched
-  Bump up version in ``src/__init__.py`` on the baseline branch to start work towards
   the next release