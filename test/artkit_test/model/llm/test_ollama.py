from collections.abc import Callable, Coroutine, Generator
from typing import Any, TypeVar, cast
from unittest.mock import AsyncMock, patch

import pytest

from artkit.model.llm.ollama import OllamaChat
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
def mock_ollama_api() -> Generator[None, None, None]:
    """Mock Ollama API validation and disable retries."""

    with (
        patch(
            "artkit.model.util._retry.retry_with_exponential_backoff",
            no_retry_decorator,
        ),
        patch(
            "artkit.model.util._retry.retry_function_with_exponential_backoff",
            no_retry_decorator,
        ),
        patch(
            "artkit.model.llm.ollama._ollama.AsyncOpenAI", autospec=True
        ) as mock_get_client,
    ):

        # Ensure `AsyncOpenAI` correctly triggers `RateLimitException`
        mock_openai_instance = mock_get_client.return_value
        mock_openai_instance.chat = AsyncMock()
        mock_openai_instance.chat.completions = AsyncMock()
        mock_openai_instance.chat.completions.create.side_effect = RateLimitException(
            "Mocked Rate Limit Exception"
        )

        yield


@pytest.mark.asyncio
async def test_ollama_retry(
    mock_ollama_api: None,
    ollama_url: str,
    ollama_model_id: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test OllamaChat handles rate limit retries correctly."""

    _ = mock_ollama_api  # Ensure the fixture is executed

    ollama_chat = OllamaChat(model_id=ollama_model_id, ollama_url=ollama_url)

    with pytest.raises(RateLimitException):
        await ollama_chat.get_response(
            "What color is the sky? Please answer in one word."
        )

    # Ensure no retries happened by checking log count
    retry_logs = [
        record
        for record in caplog.records
        if record.message.startswith("Rate limit exceeded")
    ]
    assert len(retry_logs) == 0  # There should be no retries
