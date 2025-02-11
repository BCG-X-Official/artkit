from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError

from artkit.model.llm.vllm import VLLMChat
from artkit.model.util import RateLimitException

_ = pytest.importorskip("google.generativeai")


@pytest.fixture
def mock_vllm_api() -> Generator[None, None, None]:
    """Mock both the vLLM API validation call and response generation."""

    # Mock `_validate_chat_endpoint_and_payload` so it does NOT make API requests
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "blue"}}]
        }
        mock_post.return_value = mock_response

        # Mock OpenAI client response inside VLLMChat
        with patch("artkit.model.llm.vllm._vllm.AsyncOpenAI") as mock_get_client:
            mock_openai_response = AsyncMock(
                return_value=AsyncMock(
                    choices=[
                        MagicMock(message=MagicMock(content="blue", role="assistant"))
                    ]
                )
            )
            mock_get_client.return_value.chat.completions.create = mock_openai_response

            yield  # Allows the test to use this fixture


@pytest.mark.asyncio
async def test_vllm_retry(
    mock_vllm_api: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test VLLMChat handles rate limit retries correctly."""

    # 🚀 Ensure the mock is explicitly triggered
    _ = mock_vllm_api

    # Mock OpenAI client BEFORE instantiating VLLMChat
    with (
        patch("artkit.model.llm.vllm._vllm.AsyncOpenAI") as mock_get_client,
        patch("requests.post") as mock_post,
    ):

        # Mock API validation to always succeed
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "blue"}}]
        }

        # Mock OpenAI rate limit response
        response = MagicMock()
        response.status_code = 429
        mock_get_client.return_value.chat.completions.create.side_effect = (
            RateLimitError(
                message="Rate Limit exceeded",
                response=response,
                body=MagicMock(),
            )
        )

        # Instantiate AFTER all mocks are applied
        vllm_chat = VLLMChat(model_id="gpt-3.5-turbo", vllm_url="http://localhost:8000")

        with pytest.raises(RateLimitException):
            await vllm_chat.get_response(
                "What color is the sky? Please answer in one word."
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
