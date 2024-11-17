from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError

from artkit.model.llm.vllm import VLLMChat
from artkit.model.util import RateLimitException

_ = pytest.importorskip("google.generativeai")


@pytest.mark.asyncio
async def test_vllm(vllm_chat: VLLMChat) -> None:
    # Mock openai Client
    with patch("artkit.model.llm.vllm._vllm.AsyncOpenAI") as mock_get_client:
        # Mock openai Client response
        mock_response = AsyncMock(
            return_value=AsyncMock(
                choices=[MagicMock(message=MagicMock(content="blue", role="assistant"))]
            )
        )

        # Set mock response as return value
        mock_get_client.return_value.chat.completions.create = mock_response

        # Call mocked model
        messages = await vllm_chat.get_response(
            message="What color is the sky? Please answer in one word."
        )
        assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_vllm_retry(
    vllm_chat: VLLMChat, caplog: pytest.LogCaptureFixture
) -> None:
    # Mock openai Client
    with patch("artkit.model.llm.vllm._vllm.AsyncOpenAI") as mock_get_client:
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
            await vllm_chat.get_response(
                message="What color is the sky? Please answer in one word."
            )
        assert (
            mock_get_client.return_value.chat.completions.create.call_count
            == vllm_chat.max_retries
        )
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == vllm_chat.max_retries
    )
