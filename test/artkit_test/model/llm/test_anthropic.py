import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import RateLimitError

from artkit.model.llm import CachedChatModel
from artkit.model.llm.anthropic import AnthropicChat
from artkit.model.llm.base import ChatModel
from artkit.model.util import RateLimitException

anthropic = pytest.importorskip("anthropic")


@pytest.mark.asyncio
async def test_anthropic(anthropic_chat: AnthropicChat) -> None:
    # Mock anthropic Client
    with patch(
        "artkit.model.llm.anthropic._anthropic.AsyncAnthropic"
    ) as mock_get_client:
        # Mock anthropic Client response
        mock_response = AsyncMock(
            return_value=AsyncMock(content=[MagicMock(text="blue")], role="assistant")
        )

        # Set mock response as return value
        mock_get_client.return_value.messages.create = mock_response

        # Call mocked model
        messages = await anthropic_chat.get_response(
            message="What color is the sky? Please answer in one word."
        )
        assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_cached_anthropic(
    cached_anthropic: ChatModel,
) -> None:
    messages = await cached_anthropic.get_response(
        message="What color is the sky? Please answer in one word."
    )
    assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_anthropic_retry(
    anthropic_chat: AnthropicChat, caplog: pytest.LogCaptureFixture
) -> None:
    with patch(
        "artkit.model.llm.anthropic._anthropic.AsyncAnthropic"
    ) as MockClientSession:

        # Response mock
        response = AsyncMock()
        response.status_code = 429

        MockClientSession.return_value.messages.create.side_effect = RateLimitError(
            message="Rate Limit exceeded",
            response=response,
            body=AsyncMock(),
        )

        with pytest.raises(RateLimitException):
            await anthropic_chat.with_system_prompt(
                "Your job is to answer a quiz question with a single word, "
                "lowercase, with no punctuation"
            ).get_response(message="What color is the sky?")
        assert (
            MockClientSession.return_value.messages.create.call_count
            == anthropic_chat.max_retries
        )
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == anthropic_chat.max_retries
    )


@pytest.fixture
def anthropic_chat() -> AnthropicChat:
    api_key_env = "TEST"
    os.environ[api_key_env] = "test"

    return AnthropicChat(
        max_tokens=10,
        temperature=1.0,
        model_id="claude-3-opus-20240229",
        api_key_env=api_key_env,
        initial_delay=0.1,
        exponential_base=1.5,
        max_retries=2,
    )


@pytest.fixture
def cached_anthropic(
    anthropic_chat: AnthropicChat, data_path: Path
) -> Iterator[ChatModel]:
    database = data_path / "_copy_anthropic.db"
    shutil.copyfile(data_path / "anthropic.db", database)

    yield CachedChatModel(model=anthropic_chat, database=database)

    os.remove(database)
