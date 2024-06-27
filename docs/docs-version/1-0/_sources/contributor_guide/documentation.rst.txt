.. _contributor-guide-documentation:

Documentation
=============

This section provides an overview of documentation standards for ARTKIT, including
the README, Sphinx, and docstrings.


README
------

The README file for the repo is formatted as ``.rst`` instead of the traditional markdown format. 
This facilitates incorporation of README content into the Home page of the documentation 
during the Sphinx build, which minimizes code duplication and ensures updates to our README are
reflected in the Sphinx documentation.


Sphinx Build
------------

Documentation for ARTKIT is built using `sphinx <https://www.sphinx-doc.org/en/master/>`_.
Before building the documentation ensure the ``artkit-develop`` environment is active as
the documentation build has a number of key dependencies specified in the
``environment.yml`` file, specifically:

- ``sphinx``
- ``pydata-sphinx-theme``
- ``nbsphinx``
- ``sphinx-autodoc-typehints``

To generate the Sphinx documentation locally, navigate to ``/sphinx`` and run

.. code-block:: sh

    ./make.py html

By default this will clean any previous build. The generated Sphinx
documentation for ARTKIT can then be found at ``sphinx/build/html``.

Documentation versioning is managed via the release process -- see the section on
:ref:`Package Builds <contributor-guide-builds>`.

Sphinx Folder Structure
~~~~~~~~~~~~~~~~~~~~~~~

The ``sphinx`` folder in the root directory contains the following:

- a ``make.py`` script for executing the documentation build via python

- a ``source`` directory containing predefined .rst files for the documentation build
  and other required elements (see below for more details)

- a ``base`` folder which contains

  * the ``make_base.py`` and ``conf_base.py`` scripts with nearly all configuration for
    ``make.py`` and ``conf.py``
  * ``_static`` directory, containing logos, icons, javascript and css used for documentation builds
  * ``_templates`` directory, containing *autodoc* templates used in generating and
    formatting the modules and classes for the API documentation


The ``sphinx/source`` folder contains:

- a ``conf.py`` script that is the
  `build configuration file <https://www.sphinx-doc.org/en/master/usage/configuration.html>`_
  needed to customize the input and output behavior of the Sphinx documentation build
  (see below for further details)

- the essential files used for the documentation build, which are:

  * ``index.rst``: definition of the high-level documentation structure which mainly
    references the other ``.rst`` files in this directory

  * ``user_guide/*``: pages containing a detailed user guide for ``artkit``.

  * ``examples/*``: pages containing end-to-end examples of using ``artkit`` for specific use cases.

  * ``contributor_guide/*``: pages containing detailed information on contributing to ``artkit``.

  * ``faqs.rst``: contains answers to frequently asked questions

  * ``api_landing.rst``: contains the API landing page preamble. 
    This information will appear on the API landing page in the
    documentation build after the description in ``src/__init__.py``. This
    is incorporated in the documentation build via the ``custom-module-template.rst``

- ``_static`` contains additional material used in the documentation build, in this
  case, logos and icons


The two key scripts are ``make.py`` and ``conf.py``. 

make.py
~~~~~~~

All base configuration comes from ``artkit/sphinx/make/make_base.py``, which 
includes defined commands for key steps in the documentation build. Briefly,
the key steps for the documentation build are:

- **Clean**: remove the existing documentation build

- **FetchPkgVersions**: fetch the available package versions with documentation

- **ApiDoc**: generate API documentation from sources

- **Html**: run Sphinx build to generate HTML documentation

The two other commands are **Help** and **PrepareDocsDeployment**.

conf.py
~~~~~~~

All base configuration comes from ``artkit/sphinx/make/conf_base.py``. This
`build configuration file <https://www.sphinx-doc.org/en/master/usage/configuration.html>`_
is a requirement of Sphinx and is needed to customize the input and output behavior of
the documentation build. In particular, this file highlights key extensions needed in
the build process, of which some key ones are as follows:

- `intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_
  (external links to other documentations built with Sphinx: matplotlib, numpy, ...)

- `viewcode <https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html>`_
  to include source code in the documentation, and links to the source code from the objects documentation

- `imgmath <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`_
  to render math expressions in doc strings. Note that a local latex installation is
  required (e.g., `MiKTeX <https://miktex.org/>`_ for Windows)


Docstrings
----------

The API documentation is generated from docstrings in the source code. Before writing
your own, take some time to study the existing code documentation and emulate the same
style. Describe not only what the code does, but also why, including the rationale for
any design choices that may not be obvious. Provide examples wherever this helps
explain usage patterns.

- A docstring is mandatory for all of the following entities in the source code,
  except when they are protected/private (i.e. the name starts with a leading `_`
  character):

  - modules

  - classes

  - functions/methods

  - properties

  - attributes

- Docstrings are not necessary for non-public methods, but you should have a comment
  that describes what the method does.

- Docstrings must use *reStructuredText* syntax, the default syntax for Sphinx.

- Write docstrings for functions and methods in the imperative style, e.g.,

  .. code-block:: python

      def fit():
      """Fit the model."""

  but not

  .. code-block:: python

      def fit():
      """This is a function that fits the model."""

  which is too wordy and not imperative.


- Write docstrings for modules, classes, modules, and attributes starting with a
  descriptive phrase (as you would expect in a dictionary entry). Be concise and avoid
  unnecessary or redundant phrases.
  For example:

  .. code-block:: python

      class Inspector:
          """
          Explains the inner workings of a predictive model using the SHAP approach.

          The inspector offers the following analyses:

          - ...
          - ...

  but not

  .. code-block:: python

      class Inspector:
          """
          This is a class that provides the functionality to inspect models
          ...

  as this is too verbose, and explains the class in terms of its name which does not add
  any information.

- Properties should be documented as if they were attributes, not as methods, e.g.,

  .. code-block:: python

      @property
      def children(self) -> Foo:
          """The child nodes of the tree."""
          pass

  but not

  .. code-block:: python

      @property
      def foo(self) -> Foo:
          """:return: the foo object"""
          pass

- Start full sentences and phrases with a capitalised word and end each sentence with
  punctuation, e.g.,

  .. code-block:: python

    """Fit the model."""

  but not

  .. code-block:: python

    """fit the model"""


- For multi-line docstrings, insert a line break after the leading triple quote and before
  the trailing triple quote, e.g.,

  .. code-block:: python

    def fit():
        """
        Fit the model.

        Use the underlying estimator's ``fit`` method
        to fit the model using the given training sample.

        :param sample: training sample
        """

  but not

  .. code-block:: python

    def fit():
        """Fit the model.

        Use the underlying estimator's ``fit`` method
        to fit the model using the given training sample.

        :param sample: training sample"""

- For method arguments, return value, and class parameters, one must hint the type using
  the typing module. Do not specify the parameter types in the docstrings, e.g.,

  .. code-block:: python

    def f(x: int) -> float:
      """
      Do something.

      :param x: input value
      :return: output value
      """

  but not

  .. code-block:: python

    def f(x: int) -> float:
      """
      Do something.

      :param int x: input value
      :return float: output value
      """
