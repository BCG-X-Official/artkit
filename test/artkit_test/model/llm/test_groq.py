import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from groq import RateLimitError

from artkit.model.llm import CachedChatModel
from artkit.model.llm.base import ChatModel
from artkit.model.llm.groq import GroqChat
from artkit.model.util import RateLimitException

groq = pytest.importorskip("groq")


@pytest.mark.asyncio
async def test_groq(groq_chat: GroqChat) -> None:
    # Mock groq Client
    with patch("artkit.model.llm.groq._groq.AsyncGroq") as mock_get_client:
        # Mock groq Client response
        mock_response = AsyncMock(
            return_value=AsyncMock(
                choices=[MagicMock(message=MagicMock(content="blue", role="assistant"))]
            )
        )

        # Set mock response as return value
        mock_get_client.return_value.chat.completions.create = mock_response

        # Call mocked model
        messages = await groq_chat.get_response(
            message="What color is the sky? Please answer in one word."
        )
        assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_cached_groq(
    cached_groq: ChatModel,
) -> None:
    messages = await cached_groq.get_response(
        message="What color is the sky? Please answer in one word."
    )
    assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_groq_retry(
    groq_chat: GroqChat, caplog: pytest.LogCaptureFixture
) -> None:
    with patch("artkit.model.llm.groq._groq.AsyncGroq") as MockClientSession:

        # Response mock
        response = AsyncMock()
        response.status_code = 429

        MockClientSession.return_value.chat.completions.create.side_effect = (
            RateLimitError(
                message="Rate Limit exceeded",
                response=response,
                body=AsyncMock(),
            )
        )

        with pytest.raises(RateLimitException):
            await groq_chat.with_system_prompt(
                "Your job is to answer a quiz question with a single word, "
                "lowercase, with no punctuation"
            ).get_response(message="What color is the sky?")
        assert MockClientSession.return_value.chat.completions.create.call_count == 2
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == groq_chat.max_retries
    )


@pytest.fixture
def groq_chat() -> GroqChat:
    api_key_env = "TEST"
    os.environ[api_key_env] = "test"
    return GroqChat(
        max_tokens=10,
        temperature=1.0,
        model_id="mixtral-8x7b-32768",
        api_key_env=api_key_env,
        initial_delay=0.1,
        exponential_base=1.5,
        max_retries=2,
    )


@pytest.fixture
def cached_groq(data_path: Path, groq_chat: GroqChat) -> Iterator[ChatModel]:
    database = data_path / "_copy_groq.db"
    shutil.copyfile(data_path / "groq.db", database)

    yield CachedChatModel(model=groq_chat, database=database)

    os.remove(database)
