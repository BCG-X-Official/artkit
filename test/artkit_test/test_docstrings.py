import logging

from pytools.api import DocValidator

logging.basicConfig(level=logging.INFO)


def test_docstrings() -> None:
    """
    Test all classes and functions for valid docstrings.
    """

    assert DocValidator(
        root_dir="src", exclude_from_parameter_validation=r"^artkit\.api\."
    ).validate_doc(), "docstrings are valid"
