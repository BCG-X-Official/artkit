from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError

from artkit.model.llm.vllm import VLLMChat
from artkit.model.util import RateLimitException

_ = pytest.importorskip("google.generativeai")


@pytest.fixture
def mock_vllm_api() -> Generator[None, None, None]:
    """Mock vLLM API validation and request handling."""

    with (
        patch(
            "artkit.model.llm.vllm._vllm.VLLMChat._validate_chat_endpoint_and_payload",
            autospec=True,
        ) as _mock_validate,
        patch(
            "requests.post",
            autospec=True,
            return_value=MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"role": "assistant", "content": "blue"}}]
                },
            ),
        ) as _mock_requests,
        patch(
            "artkit.model.llm.vllm._vllm.AsyncOpenAI", autospec=True
        ) as mock_get_client,
    ):

        # Ensure mock has `.chat.completions.create`
        mock_openai_instance = mock_get_client.return_value
        mock_openai_instance.chat = AsyncMock()
        mock_openai_instance.chat.completions = AsyncMock()
        mock_openai_instance.chat.completions.create = AsyncMock(
            return_value=AsyncMock(
                choices=[MagicMock(message=MagicMock(content="blue", role="assistant"))]
            )
        )

        # Suppress Pylance warnings by explicitly referencing the mocks
        _ = _mock_validate, _mock_requests

        yield  # Ensures the mocks remain active for the test


@pytest.mark.asyncio
async def test_vllm_retry(
    mock_vllm_api: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test VLLMChat handles rate limit retries correctly."""

    # Ensure the mock is explicitly triggered
    _ = mock_vllm_api

    # Mock OpenAI client BEFORE instantiating VLLMChat
    with patch("artkit.model.llm.vllm._vllm.AsyncOpenAI") as mock_get_client:

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
