from collections.abc import Callable, Coroutine, Generator
from typing import Any, TypeVar, cast
from unittest.mock import AsyncMock, patch

import pytest

from artkit.model.llm.vllm import VLLMChat
from artkit.model.util import RateLimitException

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


def no_retry_decorator(
    func: F | None = None,
    *,
    max_retries: int = 10,
    delay: int = 1,
    exponential_base: int = 2,
    jitter: bool = True,
) -> F:
    """
    Mock retry decorator that properly handles expected parameters but disables retries.
    """

    def decorator(inner_func: F) -> F:
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return await inner_func(self, *args, **kwargs)

        return cast(F, wrapper)  # ✅ Correct return type enforcement

    if func is not None:
        return decorator(func)

    return cast(F, decorator)


@pytest.fixture
def mock_vllm_api() -> Generator[None, None, None]:
    """Mock vLLM API validation and disable retries."""

    with (
        patch(
            "artkit.model.llm.vllm._vllm.VLLMChat._validate_chat_endpoint_and_payload",
            autospec=True,
        ) as mock_validate,
        patch(
            "artkit.model.util._retry.retry_with_exponential_backoff",
            no_retry_decorator,
        ),
        patch(
            "artkit.model.util._retry.retry_function_with_exponential_backoff",
            no_retry_decorator,
        ),
        patch(
            "artkit.model.llm.vllm._vllm.AsyncOpenAI", autospec=True
        ) as mock_get_client,
    ):
        mock_validate.return_value = None

        # Ensure `AsyncOpenAI` correctly triggers `RateLimitException`
        mock_openai_instance = mock_get_client.return_value
        mock_openai_instance.chat = AsyncMock()
        mock_openai_instance.chat.completions = AsyncMock()
        mock_openai_instance.chat.completions.create.side_effect = RateLimitException(
            "Mocked Rate Limit Exception"
        )

        yield


@pytest.mark.asyncio
async def test_vllm_retry(
    mock_vllm_api: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test VLLMChat handles rate limit retries correctly."""

    _ = mock_vllm_api  # Ensure the fixture is executed

    vllm_chat = VLLMChat(model_id="gpt-3.5-turbo", vllm_url="http://localhost:8000")

    with pytest.raises(RateLimitException):
        await vllm_chat.get_response(
            "What color is the sky? Please answer in one word."
        )

    # Ensure no retries happened by checking log count
    retry_logs = [
        record
        for record in caplog.records
        if record.message.startswith("Rate limit exceeded")
    ]
    assert len(retry_logs) == 0  # There should be no retries
