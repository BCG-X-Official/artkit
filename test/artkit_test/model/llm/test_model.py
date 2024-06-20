"""
General tests for the 'model' module.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pytest

from artkit.model.base import ConnectorMixin

log = logging.getLogger(__name__)


def test_api_key() -> None:

    # A dummy model
    class MyModel(ConnectorMixin[Any]):

        @classmethod
        def get_default_api_key_env(cls) -> str:
            return "MY_KEY"

        def _make_client(self) -> Any:
            pass

    # Create a dummy API key in the environment
    os.environ["MY_CUSTOM_KEY"] = "my_custom_key"

    # Create the model with a custom API key environment variable
    model = MyModel(model_id="my_model", api_key_env="MY_CUSTOM_KEY")

    # Test the API key
    assert model.get_api_key() == "my_custom_key"

    # Create a dummy API key in the environment
    os.environ["MY_KEY"] = "my_key"

    # Create the model with the default API key environment variable
    model = MyModel(model_id="my_model")

    # Test the API key
    assert model.get_api_key() == "my_key"

    # Removing the dummy API key leads to an error when trying to get the API key
    del os.environ["MY_KEY"]
    with pytest.raises(
        ValueError,
        match=(
            r"^The environment variable MY_KEY for the API key of model 'my_model' is "
            r"not set\. Please set the environment variable to your API key, or revise "
            r"arg api_key_env to the correct environment variable name\.$"
        ),
    ):
        model.get_api_key()
