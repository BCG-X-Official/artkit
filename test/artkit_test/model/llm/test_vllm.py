from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError

from artkit.model.llm.vllm import VLLMChat
from artkit.model.util import RateLimitException

_ = pytest.importorskip("google.generativeai")


@pytest.mark.asyncio
async def test_vllm() -> None:
    # Mock openai Client
    with patch("artkit.model.llm.vllm._vllm.AsyncOpenAI") as mock_get_client:
        # Mock OpenAI Client response
        mock_response = AsyncMock(
            return_value=AsyncMock(
                choices=[MagicMock(message=MagicMock(content="blue", role="assistant"))]
            )
        )

        # Apply the mock response
        mock_get_client.return_value.chat.completions.create = mock_response

        # Instantiate VLLMChat AFTER applying the mock
        vllm_chat = VLLMChat(model_id="gpt-3.5-turbo", vllm_url="http://localhost:8000")

        # Call mocked model
        messages = await vllm_chat.get_response(
            message="What color is the sky? Please answer in one word."
        )
        assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_vllm_retry(caplog: pytest.LogCaptureFixture) -> None:
    # Mock openai Client
    with patch("artkit.model.llm.vllm._vllm.AsyncOpenAI") as mock_get_client:
        # Set up a mock response that triggers a rate limit error
        response = MagicMock()
        response.status_code = 429

        mock_get_client.return_value.chat.completions.create.side_effect = (
            RateLimitError(
                message="Rate Limit exceeded",
                response=response,
                body=MagicMock(),
            )
        )

        # Instantiate VLLMChat AFTER applying the mock
        vllm_chat = VLLMChat(model_id="gpt-3.5-turbo", vllm_url="http://localhost:8000")

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
