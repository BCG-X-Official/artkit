import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core.exceptions import TooManyRequests

from artkit.model.llm import CachedChatModel
from artkit.model.llm.base import ChatModel
from artkit.model.llm.gemini import GeminiChat
from artkit.model.util import RateLimitException

_ = pytest.importorskip("google.generativeai")


@pytest.mark.asyncio
async def test_gemini(gemini_chat: GeminiChat) -> None:
    # Mock Gemini Client
    with patch("artkit.model.llm.gemini._gemini.GenerativeModel") as mock_get_client:
        # Mock Gemini Client response
        mock_response = AsyncMock(
            return_value=AsyncMock(
                candidates=[
                    MagicMock(content=MagicMock(parts=[MagicMock(text="blue")]))
                ]
            )
        )

        # Set mock response as return value
        mock_get_client.return_value.generate_content_async = mock_response

        # Call mocked model
        messages = await gemini_chat.get_response(
            message="What color is the sky? Please answer in one word."
        )
        assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_gemini_retry(
    gemini_chat: GeminiChat, caplog: pytest.LogCaptureFixture
) -> None:
    # Mock Gemini Client
    with patch("artkit.model.llm.gemini._gemini.GenerativeModel") as mock_get_client:
        # Set mock response as return value
        mock_get_client.return_value.generate_content_async.side_effect = (
            TooManyRequests("Rate limit error")
        )

        with pytest.raises(RateLimitException):
            # Call mocked model
            await gemini_chat.get_response(
                message="What color is the sky? Please answer in one word."
            )
        assert (
            mock_get_client.return_value.generate_content_async.call_count
            == gemini_chat.max_retries
        )
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == gemini_chat.max_retries
    )


@pytest.mark.asyncio
async def test_cached_gemini(
    cached_gemini: ChatModel,
) -> None:
    messages = await cached_gemini.get_response(
        message="What color is the sky? Please answer in one word."
    )
    assert "blue" in messages[0].lower()


@pytest.fixture
def gemini_chat() -> GeminiChat:
    api_key_env = "TEST"
    os.environ[api_key_env] = "test"

    return GeminiChat(
        model_id="models/gemini-pro",
        api_key_env=api_key_env,
        max_output_tokens=10,
        max_retries=2,
        initial_delay=0.1,
        exponential_base=1.5,
    )


@pytest.fixture
def cached_gemini(data_path: Path, gemini_chat: GeminiChat) -> Iterator[ChatModel]:
    database = data_path / "_copy_gemini.db"
    shutil.copyfile(data_path / "gemini.db", database)

    yield CachedChatModel(model=gemini_chat, database=database)

    os.remove(database)
