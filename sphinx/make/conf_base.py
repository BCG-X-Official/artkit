"""
Configuration file for the Sphinx documentation builder.

For a full list of options see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import logging
import os
import shutil
import sys
from typing import Any

from sphinx.application import Sphinx

from pytools.sphinx.util import (
    Replace3rdPartyDoc,
    ResolveTypeVariables,
    UpdateForwardReferences,
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(name=__name__)

# this is the directory that contains all required repos
_dir_conf_base = os.path.dirname(os.path.abspath(__file__))
_dir_sphinx = os.path.dirname(_dir_conf_base)
_dir_src = os.path.join(os.path.dirname(_dir_sphinx), "src")


# noinspection PyShadowingNames
def set_config(
    globals_: dict[str, Any],
    *,
    project: str,
    html_logo: str | None = None,
    intersphinx_mapping: dict[str, tuple[str, str | None]] | None = None,
) -> None:
    """
    Add required modules to the python path, and set custom configuration options
    """

    from make_util import get_package_version

    globals_["project"] = project
    globals_["version"] = str(
        get_package_version(package_path=os.path.join(_dir_src, project))
    )

    globals_["html_show_sourcelink"] = False

    if html_logo:
        globals_["html_logo"] = html_logo
        globals_["latex_logo"] = html_logo

    if intersphinx_mapping:
        _update_intersphinx_mapping(intersphinx_mapping)

    module_path = _dir_src
    if module_path not in sys.path:
        # noinspection PyUnboundLocalVariable
        sys.path.insert(0, module_path)
        _log.info(f"added '{module_path}' to python paths")

    # Update global variables
    globals_.update(
        (k, v) for k, v in globals().items() if not (k.startswith("_") or k in globals_)
    )


def _update_intersphinx_mapping(mapping: dict[str, tuple[str, str | None]]) -> None:
    global intersphinx_mapping
    intersphinx_mapping.update(mapping)


_log.info(f"sys.path = {sys.path}")

# -- Project information -----------------------------------------------------

# noinspection PyShadowingBuiltins
copyright = "2024, Boston Consulting Group (BCG)"
author = "BCG X Responsible AI Team"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# -- Options for autodoc / autosummary -------------------------------------------------

# prevent numpydoc from interfering with autosummary
numpydoc_show_class_members = False

# generate autosummary even if no references
autosummary_generate = False

# always overwrite generated auto summaries with newly generated versions
autosummary_generate_overwrite = True

autosummary_imported_members = True

autodoc_default_options = {
    "ignore-module-all": False,
    "inherited-members": True,
    "imported-members": False,
    "show-inheritance": False,
    "member-order": "groupwise",
}

nbsphinx_allow_errors = True

# add intersphinx mapping
intersphinx_mapping: dict[str, tuple[str, str | None]] = {
    "fluxus": ("https://bcg-x-official.github.io/fluxus", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pd": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3.7", None),
}

intersphinx_collapsible_submodules = {
    "pandas.core.frame": "pandas",
    "pandas.core.series": "pandas",
    "pandas.core.indexes.base": "pandas",
    "pandas.core.indexes.multi": "pandas",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["*/.ipynb_checkpoints/*", "**/supplementary/**"]

# -- Options for sphinx_autodoc_typehints ----------------------------------------------
set_type_checking_flag = False
typehints_fully_qualified = False
always_document_param_types = False
autodoc_typehints_format = "short"

# -- Options for Math output -----------------------------------------------------------

imgmath_image_format = "svg"
imgmath_use_preview = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/BCG-X-Official/artkit",
            "icon": "fab fa-github",
        },
    ],
    "collapse_navigation": True,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 6,
    # Omit the `theme-switcher` since we don't support dark mode
    # (as of pydata theme 0.9):
    "navbar_end": ["navbar-icon-links"],
    # "secondary_sidebar_items": {"path/to/page": [],},
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "show_prev_next": False,
}

html_sidebars: dict[str, list[str]] = {
    "_generated/home": [],
    "faq": [],
    "_generated/release_notes": [],
}

# Set the default mode to "light" instead of "auto" (as of pydata theme 0.9):
html_context = {"default_mode": "light"}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../make/_static"]
html_logo = os.path.join("_static", "bcgx_logo.jpg")

# Class documentation to include docstrings both global to the class, and from __init__
autoclass_content = "both"

# -- End of options section ------------------------------------------------------------


def setup(app: Sphinx) -> None:
    """
    Add event handlers to the Sphinx application object.

    :param app: the Sphinx application object
    """

    _connect_callbacks(app=app)

    _add_custom_css_and_js(app=app)


def _connect_callbacks(app: Sphinx) -> None:
    from pytools.sphinx.util import (
        AddInheritance,
        CollapseModulePathsInDocstring,
        CollapseModulePathsInSignature,
        CollapseModulePathsInXRef,
        RenamePrivateArguments,
        SkipIndirectImports,
        TrackCurrentClass,
    )

    UpdateForwardReferences().connect(app=app)

    AddInheritance(collapsible_submodules=intersphinx_collapsible_submodules).connect(
        app=app
    )

    CollapseModulePathsInDocstring(
        collapsible_submodules=intersphinx_collapsible_submodules
    ).connect(app=app, priority=100000)

    CollapseModulePathsInSignature(
        collapsible_submodules=intersphinx_collapsible_submodules
    ).connect(app=app, priority=100000)

    SkipIndirectImports().connect(app=app)

    Replace3rdPartyDoc().connect(app=app)

    TrackCurrentClass().connect(app=app)

    ResolveTypeVariables().connect(app=app)

    CollapseModulePathsInXRef(
        collapsible_submodules=intersphinx_collapsible_submodules
    ).connect(app=app)

    RenamePrivateArguments().connect(app=app)


def _add_custom_css_and_js(app: Sphinx) -> None:
    # add custom css and js files, and copy them to the build/html/_static folder
    css_rel_paths = [os.path.join("css", "bcgx.css")]
    js_rel_paths = [os.path.join("js", "bcgx.js"), os.path.join("js", "versions.js")]

    for css in css_rel_paths:
        app.add_css_file(filename=css)

    for js in js_rel_paths:
        app.add_js_file(filename=js)

    src_root = os.path.join(_dir_sphinx, "make", "_static")
    dst_html = os.path.join(_dir_sphinx, "build", "html", "_static")
    for rel_path in css_rel_paths + js_rel_paths:
        dst_dir = os.path.join(dst_html, os.path.dirname(rel_path))
        os.makedirs(dst_dir, exist_ok=True)
        src_file = os.path.join(src_root, rel_path)
        _log.info(f"copying {src_file} to {dst_dir}")
        shutil.copy(src=src_file, dst=dst_dir)
