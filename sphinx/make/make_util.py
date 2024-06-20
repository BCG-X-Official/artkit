"""
Utility functions for sphinx and package builds.
"""

import os
import re
import sys

from packaging import version as pkg_version

RE_VERSION_DECLARATION = re.compile(r"\b__version__\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")


def get_package_version(package_path: str) -> pkg_version.Version:
    """
    Retrieve the package version for the project from __init__

    :param package_path: the root directory of the package
    """
    init_path = os.path.abspath(os.path.join(package_path, "__init__.py"))

    print(f"Retrieving package version from {init_path}", file=sys.stderr)

    with open(init_path) as init_file:
        init_lines = init_file.readlines()

    matches = {
        match[1] or match[2]
        for match in (RE_VERSION_DECLARATION.match(line) for line in init_lines)
        if match
    }

    if len(matches) == 0:
        raise RuntimeError(f"No valid __version__ declaration found in {init_path}")

    elif len(matches) > 1:
        raise RuntimeError(
            f"Multiple conflicting __version__ declarations found in {init_path}: "
            f"{matches}"
        )

    else:
        package_version = next(iter(matches))

    return pkg_version.parse(package_version)
