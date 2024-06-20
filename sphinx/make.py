#!/usr/bin/env python3
"""
Make sphinx documentation using the makefile in the ``make`` subdirectory
"""

import os
import sys

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "make")
    )

    from make_base import make

    make()
