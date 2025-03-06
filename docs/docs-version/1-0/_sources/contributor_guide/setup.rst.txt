.. _contributor-guide-setup:

Setup
=====

This section describes the steps to setting up a development environment for ARTKIT.

Pre-requisites
~~~~~~~~~~~~~~

The basic requirements for developing this library are:

-  `Python <https://www.python.org/downloads/>`__ 3.10 or later
-  `git <https://git-scm.com/downloads>`__ for cloning and contributing to the library
-  `pip <https://pip.pypa.io/en/stable/installation/>`__ or `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__ for installing and managing Python libraries

We recommend using an IDE such as `VS
Code <https://code.visualstudio.com/>`__ or
`PyCharm <https://www.jetbrains.com/pycharm/>`__.

Clone git repository
~~~~~~~~~~~~~~~~~~~~~

Clone a local version of the library using HTTPS:

::

   git clone https://github.com/bcgx-pi-PID-XGF-08/artkit.git

or SSH:

::

   git clone git@github.com:bcgx-pi-PID-XGF-08/artkit.git

Configure environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

API credentials should be defined as environment variables in a ``.env``
file in the project root. This file is listed in ``.gitignore`` to `keep
your credentials secure <https://blog.gitguardian.com/secrets-api-management/>`__. 
ARTKIT includes a template file called ``.env_example`` in the project root with 
placeholder credentials for supported APIs. Modify this file with your credentials
and rename it to ``.env``.

To load environment variables in a script or notebook, use:

::

   from dotenv import load_dotenv
   load_dotenv()

| To encourage secure storage of credentials, ARTKIT classes which make
  API requests do not accept API credentials directly, but instead
  require environmental variables to be defined.
| For example, if your OpenAI API key is stored in an environment
  variable called ``OPENAI_API_KEY``, then you can then initialize
  objects that interact with OpenAI API like this:

::

   from artkit.api import OpenAIChat

   llm = OpenAIChat(api_key_env="OPENAI_API_KEY", ...)

Note that the ``api_key_env`` variable accepts the name of the
environment variable as a string instead of directly accepting an API
key as a parameter. This pattern reduces the risk of accidental exposure 
of API keys in code repositories, since the key is not stored as a Python 
object which can be printed. 

Create virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the project root, create a dedicated environment for your project
using, e.g., ``venv``:

::

   python -m venv artkit

and activate it with

::

   source artkit/bin/activate

on mac or ``.\artkit\scripts\Activate`` on Windows.

Enable import of local ARTKIT modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing the project locally in developer mode to enable
importing local ARTKIT modules in scripts or notebooks as if the library
is installed, but with local changes immediately reflected.

To install ARTKIT in developer mode, run the following from your
project root:

::

   pip install -e ".[dev]"

As an alternative approach, you can add the folder ``artkit/src`` to
your ``PYTHONPATH``, and this will enable importing local ARTKIT modules
into scripts or notebooks.

Install dependencies
~~~~~~~~~~~~~~~~~~~~

The following installations are required for full functionality.

GraphViz
^^^^^^^^

`GraphViz <https://graphviz.org/>`__ is required for generating pipeline
flow diagrams. Install the library and ensure it is in your system’s
PATH variable:

-  For MacOS and Linux users, simple instructions provided on `GraphViz
   Downloads <https://www.graphviz.org/download/>`__ should
   automatically add GraphViz to your path
-  Windows users may need to manually add GraphViz to your PATH (see
   `Simplified Windows installation
   procedure <https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224>`__)
-  Run ``dot -V`` in Terminal or Command Prompt to verify installation

Pandoc
^^^^^^

Pandoc is required to render Jupyter Notebooks for building the sphinx
documentation.

-  MacOS users can install Pandoc with ``brew install pandoc``
-  Windows and Linux users should follow the `Pandoc
   installation <https://pandoc.org/installing.html>`__ instructions for
   their system

Install pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~

This project uses `pre-commit hooks <https://pre-commit.com/>`__ to
automatically enforce uniform coding standards in commits:

::

   pre-commit install

To execute the pre-commit hooks on demand, use ``pre-commit run`` from
the command line.

Run unit tests
~~~~~~~~~~~~~~

This project uses `pytest <https://docs.pytest.org/en/8.0.x/>`__ to
support functional testing. To run the test suite:

::

   pytest

To maintain high standard for test coverage, the testing pipeline is
configured to require at least 90% test coverage of the codebase,
otherwise ``pytest`` will exit with a failure status.

Build documentation
~~~~~~~~~~~~~~~~~~~

Ensure the src directory does not contain any environment-specific
folders or files which can be interpreted as packages, such as
``\*.egg_info/``. If you do, you will get the following error during the
sphinx build:
``assert len(PACKAGE_NAMES) == 1, "only one package per Sphinx build is supported"``.

To build the `sphinx <https://www.sphinx-doc.org/en/master/>`__
documentation, navigate to the ``sphinx`` directory and run:

::

   ./make.py html

To view the documentation, open ``sphinx/build/html/index.html`` in the
web browser of your choice.
