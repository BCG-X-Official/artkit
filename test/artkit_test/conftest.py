import base64
import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from artkit.model.llm import CachedChatModel
from artkit.model.llm.base import ChatModel
from artkit.model.llm.openai import OpenAIChat
from artkit.util import Image

log = logging.getLogger(__name__)

DIR_TEST_ROOT = "test"
DIR_TEST_SOURCES = os.path.join(DIR_TEST_ROOT, "artkit_test")
DIR_TEST_DATA = os.path.join(DIR_TEST_ROOT, "data")
FILE_LLM_CACHE_ASYNC_JSON = os.path.join(DIR_TEST_DATA, "llm_cache_async.json")

# validate that the test sources directory exists
if not os.path.exists(DIR_TEST_SOURCES):
    raise FileNotFoundError(
        f"Test sources directory does not exist: {DIR_TEST_SOURCES}. "
        "Make sure to set the working directory to the project root directory."
    )

# create the test data directory if it does not exist
os.makedirs(DIR_TEST_DATA, exist_ok=True)


@pytest.fixture
def data_path() -> Path:
    return Path(DIR_TEST_DATA)


@pytest.fixture
def cache_db_path(data_path: Path) -> Path:
    return data_path.joinpath("cache.db")


@pytest.fixture(scope="session")
def openai_model() -> str:
    # the openai model type to use
    return "gpt-4"


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    # the name of the environment variable holding the OpenAI API key
    return "OPENAI_API_KEY"


@pytest.fixture
def openai_chat() -> OpenAIChat:
    api_key_env = "TEST"
    os.environ[api_key_env] = "test"

    return OpenAIChat(
        model_id="gpt-3.5-turbo",
        api_key_env=api_key_env,
        temperature=0.8,
        seed=0,
        max_output_tokens=10,
        max_retries=2,
        initial_delay=0.1,
        exponential_base=1.5,
    )


@pytest.fixture
def cached_openai(data_path: Path, openai_chat: OpenAIChat) -> Iterator[ChatModel]:
    database = data_path / "_copy_openai.db"
    shutil.copyfile(data_path / "openai.db", database)

    yield CachedChatModel(
        model=openai_chat,
        database=database,
    )

    os.remove(database)


@pytest.fixture
def image() -> Image:
    # The base64 string representing the PNG
    # noinspection SpellCheckingInspection
    base64_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAgcBApLtYoM"
        "AAAAASUVORK5CYII="
    )

    # Decode the base64 string
    binary_png = base64.b64decode(base64_png)

    return Image(data=binary_png)
