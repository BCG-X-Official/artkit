"""
Caching for ARTKIT model connections.

This module provides a caching mechanism for ARTKIT model connections. The
caching mechanism prevents repeated requests to the model connection, thus
reducing costs and improving performance.

This is useful not only for testing and debugging, but also for production
systems where the model connection is used frequently, e.g., for regular automated
tests that require a large number of requests to the model to generate or augment test
prompts.

The cache is implemented using Python's built-in ``sqlite3`` module, which provides a
lightweight, serverless, and self-contained SQL database engine. The cache is
stored in a single file on disk and can be easily shared between different
processes or threads, or even between different users.
"""

from ._cache import *
