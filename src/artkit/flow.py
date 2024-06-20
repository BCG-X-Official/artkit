# -----------------------------------------------------------------------------
# © 2024 Boston Consulting Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------

"""
This module provides a functional API for defining and executing
complex and asynchronous workflows. The API is inspired by the functional programming
paradigm and is designed to be simple, expressive, and composable.

It is based on the :mod:`fluxus.functional` module of the *fluxus* library.
Please refer to the *fluxus* documentation for more information.

The functions :func:`step`, :func:`chain`, and :func:`parallel` are used to
define the steps of a flow. Function :func:`step` defines a single step, function
:func:`chain` composes steps or flows sequentially, and the :func:`parallel` composes
steps or flows in parallel.

Alternatively, the ``>>`` operator can be used to chain steps or flows, and the
``&`` operator can be used to compose steps or flows in parallel.

Function :func:`passthrough` can be included in a parallel composition to pass the
unchanged input through to the next step.

Function :func:`run` is used to execute a flow. The flow is executed asynchronously,
and all results are returned as a :class:`RunResult` instance. Run results include
the input to the flow as well as all intermediate and final results along all paths
through the flow.

Example:

.. code-block:: python

    from artkit.flow import step, chain, parallel, passthrough, run
    from collections.abc import AsyncIterator
    from typing import Any

    def add_one(x) -> dict[str, Any]:
        # We can return a single dict
        return dict(x=x + 1)

    async def add_two(x) -> dict[str, Any]:
        # We can use asynchronous functions
        return dict(x=x + 2)

    async def add_three_or_five(x) -> AsyncIterator[dict[str, Any]]:
        # We can yield rather than return values to generate multiple products
        yield dict(x=x + 3)
        yield dict(x=x + 5)

    flow = chain(
        step("add_1", add_one),
        parallel(
            step("add_2", add_two),
            step("add_3_or_5", add_three_or_five),
            passthrough(),
        ),
    )

    result = run(flow, input=[dict(x=1), dict(x=10)])

    print(result)

This will output:

.. code-block:: python

    RunResult(
        [
            {'input': {'x': 1}, 'add_1': {'x': 2}, 'add_2': {'x': 4}},
            {'input': {'x': 10}, 'add_1': {'x': 11}, 'add_2': {'x': 13}}
        ],
        [
            {'input': {'x': 1}, 'add_1': {'x': 2}, 'add_3_or_5': {'x': 5}},
            {'input': {'x': 1}, 'add_1': {'x': 2}, 'add_3_or_5': {'x': 7}},
            {'input': {'x': 10}, 'add_1': {'x': 11}, 'add_3_or_5': {'x': 14}},
            {'input': {'x': 10}, 'add_1': {'x': 11}, 'add_3_or_5': {'x': 16}}
        ],
        [
            {'input': {'x': 1}, 'add_1': {'x': 2}},
            {'input': {'x': 10}, 'add_1': {'x': 11}}
        ]
    )

The following functions and classes are provided by this module:
"""


def __mirror_fluxus() -> None:  # pragma: no cover
    import os
    from typing import Any

    import fluxus.functional as fluxus

    # Get all public definitions from the fluxus module as tuples (name, obj)
    fluxus_definitions: list[tuple[str, Any]] = sorted(
        (name, obj)
        for name, obj in vars(fluxus).items()
        if not name.startswith("_") and callable(obj)
    )

    # Get the globals of this module
    globals_ = globals()

    # Generate bullet points for the module docstring
    docstring = "\n".join(
        f"- :obj:`~fluxus.functional.{name}`" for name, obj in fluxus_definitions
    )

    # Add the bullet points to the module docstring
    globals_["__doc__"] += "\n\n" + docstring

    # Check if we are building the Sphinx documentation
    if "SPHINX_BUILD" in os.environ:
        # we are building the documentation - we are done
        return

    # We are not building the Sphinx documentation - add the fluxus definitions to this
    # module

    # Add all public definitions from the fluxus module to this module
    for name, obj in fluxus_definitions:
        globals_[name] = obj

    # Set the __all__ attribute to all public symbols
    globals_["__all__"] = [name for name, _ in fluxus_definitions]


# Update the module attributes of the functions and classes imported from fluxus
__mirror_fluxus()

# Clean up the namespace
del __mirror_fluxus
