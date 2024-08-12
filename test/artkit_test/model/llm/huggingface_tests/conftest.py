import os

import pytest


@pytest.fixture
def hf_token() -> str:
    api_key = "HF_TOKEN"
    # create a dummy key if not found
    if api_key not in os.environ:
        os.environ[api_key] = "<HF API key>"
    return api_key
