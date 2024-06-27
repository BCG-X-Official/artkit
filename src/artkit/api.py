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
Convenience module to import all classes and functions using single package alias.

Recommended usage is to import this module as ``ak``.

Example:

.. code-block:: python

    import artkit.api as ak

    ak.run(
        ak.step("add", lambda x: dict(x=x + 1)),
        input=[dict(x=1), dict(x=10)],
    )

"""

__globals_original = set(globals().keys())


from .flow import *
from .model.diffusion import *
from .model.diffusion.base import DiffusionModel
from .model.diffusion.openai import *
from .model.llm import *
from .model.llm.anthropic import *
from .model.llm.base import ChatModel, CompletionModel
from .model.llm.gemini import *
from .model.llm.groq import *
from .model.llm.huggingface import *
from .model.llm.multi_turn import *
from .model.llm.openai import *
from .model.vision import *
from .model.vision.base import VisionModel
from .model.vision.openai import *
from .util import *


def __adjust_namespace_for_sphinx() -> None:  # pragma: no cover
    import os
    from collections import defaultdict

    from pytools.api import public_module_prefix

    # Remove symbols that are not functions or classes
    for name, obj in list(globals().items()):
        if not callable(obj) and not name.startswith("__"):
            del globals()[name]

    # Check if we are building the Sphinx documentation
    if "SPHINX_BUILD" not in os.environ:
        # we are not building the documentation - no adjustments needed
        return

    # We are building the Sphinx documentation

    # Get the namespace of this module
    globals_ = globals()

    # Identify all imported symbols
    imported_symbols = set(globals_.keys()) - __globals_original  # noqa: F405

    # Remove our internal symbols from items to be documented
    imported_symbols.remove("__globals_original")
    imported_symbols.remove(__adjust_namespace_for_sphinx.__name__)  # noqa: F405

    # Add all imported symbols to the module docstring
    imported_objects = [globals_[symbol] for symbol in imported_symbols]

    # Group the imported objects by their module and an optional module alias
    imported_objects_by_module: dict[tuple[str, str | None], list[str]] = defaultdict(
        list
    )
    for obj in imported_objects:
        imported_objects_by_module[(public_module_prefix(obj.__module__), None)].append(
            obj.__qualname__
        )

    # Special case: mirror the fluxus module
    import fluxus.functional as fx

    imported_objects_by_module[("fluxus.functional", "artkit.flow")] = sorted(
        (
            name
            for name, obj in vars(fx).items()
            if not name.startswith("_") and callable(obj)
        ),
        key=lambda s: s.lower(),
    )

    # Create the documentation string, with separate sections for each module
    doc = [
        f"\n* :mod:`{alias or module}`   \n\n"
        + "\n".join(f"  - :obj:`~{module}.{name}`" for name in names)
        for (module, alias), names in sorted(
            # Sort by module then by name
            imported_objects_by_module.items(),
            key=lambda x: (
                # sort by alias, if it exists, otherwise by name
                x[0][1] or x[0][0],
                x[1],
            ),
        )
    ]
    globals_["__doc__"] += (
        "Below is a list of all symbols exposed through ``artkit.api``:\n\n"
        + "\n".join(doc)
        + "\n"
    )

    # remove all imported and documented symbols from the global namespace
    for symbol in imported_symbols:
        del globals_[symbol]


# Adjust the namespace for Sphinx documentation builds
__adjust_namespace_for_sphinx()

# Clean up the namespace
del __adjust_namespace_for_sphinx, __globals_original
