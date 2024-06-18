"""
Validate the Python code in the module.
"""

from pytools.api import validate__all__declarations


def test__all__declarations() -> None:
    import artkit

    validate__all__declarations(artkit)
