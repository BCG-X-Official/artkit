"""
Configuration file for the Sphinx documentation builder.

Receives the majority of the configuration from ``conf_base.py``.
"""

import os
import sys

_dir_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "make")
sys.path.insert(0, _dir_base)

from conf_base import set_config  # noqa: E402

# ----- set custom configuration -----

set_config(
    globals(),
    project="artkit",
    html_logo=os.path.join("_images", "ARTKIT_Logo_Light_RGB-small.png"),
)
