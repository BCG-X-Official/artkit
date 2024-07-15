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
-  IMPORTANT: Update version in ``src/__init__.py`` on the baseline branch (1.0.x) to be the next minor version release (if the last release was 1.0.1, 
   the next version should be 1.0.2)