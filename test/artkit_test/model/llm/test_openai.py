from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError

from artkit.model.llm.base import ChatModel
from artkit.model.llm.openai import OpenAIChat
from artkit.model.util import RateLimitException

_ = pytest.importorskip("google.generativeai")


@pytest.mark.asyncio
async def test_openai(openai_chat: OpenAIChat) -> None:
    # Mock openai Client
    with patch("artkit.model.llm.openai._openai.AsyncOpenAI") as mock_get_client:
        # Mock openai Client response
        mock_response = AsyncMock(
            return_value=AsyncMock(
                choices=[MagicMock(message=MagicMock(content="blue", role="assistant"))]
            )
        )

        # Set mock response as return value
        mock_get_client.return_value.chat.completions.create = mock_response

        # Call mocked model
        messages = await openai_chat.get_response(
            message="What color is the sky? Please answer in one word."
        )
        assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_openai_retry(
    openai_chat: OpenAIChat, caplog: pytest.LogCaptureFixture
) -> None:
    # Mock openai Client
    with patch("artkit.model.llm.openai._openai.AsyncOpenAI") as mock_get_client:
        # Set mock response as return value
        response = MagicMock()
        response.status_code = 429

        # Mock exception on method call
        mock_get_client.return_value.chat.completions.create.side_effect = (
            RateLimitError(
                message="Rate Limit exceeded",
                response=response,
                body=MagicMock(),
            )
        )

        with pytest.raises(RateLimitException):
            # Call mocked model
            await openai_chat.get_response(
                message="What color is the sky? Please answer in one word."
            )
        assert (
            mock_get_client.return_value.chat.completions.create.call_count
            == openai_chat.max_retries
        )
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == openai_chat.max_retries
    )


@pytest.mark.asyncio
async def test_cached_openai(
    cached_openai: ChatModel,
) -> None:
    messages = await cached_openai.get_response(
        message="What color is the sky? Please answer in one word."
    )
    assert "blue" in messages[0].lower()
