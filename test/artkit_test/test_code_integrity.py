"""
Validate the Python code in the module.
"""

from pytools.api import validate__all__declarations


def test__all__declarations() -> None:
    import artkit
    import artkit.flow as flow

    # artkit.flow exports 3rd party symbols from the fluxus package, so we need
    # to set the modules to artkit.flow to make this test succeed

    flow_namespace = vars(flow)
    original_modules = {
        name: obj.__module__
        for name, obj in flow_namespace.items()
        if not name.startswith("_")
    }
    try:
        # Set the module of the symbols to artkit.flow
        for name in original_modules:
            flow_namespace[name].__module__ = flow.__name__

        validate__all__declarations(artkit)
    finally:
        # Restore the original modules
        for name, module in original_modules.items():
            flow_namespace[name].__module__ = module
