from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core.exceptions import TooManyRequests

from artkit.model.llm.vertexai._vertexai import VertexAIChat
from artkit.model.util import RateLimitException

_ = pytest.importorskip("vertexai")


@pytest.mark.asyncio
async def test_vertexai(vertex_chat: VertexAIChat) -> None:
    # Mock Vertex AI Client
    with patch(
        "artkit.model.llm.vertexai._vertexai.GenerativeModel"
    ) as mock_get_client:
        # Mock Vertex AI Client response
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
        messages = await vertex_chat.get_response(
            message="What color is the sky? Please answer in one word."
        )
        assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_vertexai_retry(
    vertex_chat: VertexAIChat, caplog: pytest.LogCaptureFixture
) -> None:
    # Mock Vertex AI Client
    with patch(
        "artkit.model.llm.vertexai._vertexai.GenerativeModel"
    ) as mock_get_client:
        # Set mock response as return value
        mock_get_client.return_value.generate_content_async.side_effect = (
            TooManyRequests("Rate limit error")
        )

        with pytest.raises(RateLimitException):
            # Call mocked model
            await vertex_chat.get_response(
                message="What color is the sky? Please answer in one word."
            )
        assert (
            mock_get_client.return_value.generate_content_async.call_count
            == vertex_chat.max_retries
        )
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == vertex_chat.max_retries
    )


FIRST_SYSTEM_PROMPT = (
    "Your name is first bot, and you are helpful, creative, clever, and very friendly."
)
SECOND_SYSTEM_PROMPT = (
    "Your name is second bot, and you are mean, snarky and misleading."
)


@pytest.mark.asyncio
async def test_vertexai_different_system_prompts() -> None:
    # Create two instances of VertexAIChat with different system prompts
    first_chat = VertexAIChat(
        model_id="gemini-1.5-pro",
        gcp_project_id="gcp-project",
    ).with_system_prompt(FIRST_SYSTEM_PROMPT)
    second_chat = VertexAIChat(
        model_id="gemini-1.5-pro",
        gcp_project_id="gcp-project",
    ).with_system_prompt(SECOND_SYSTEM_PROMPT)

    # Mock Vertex AI Client
    with patch(
        "artkit.model.llm.vertexai._vertexai.GenerativeModel"
    ) as mock_get_client:
        # Mock responses for each instance
        mock_first_response = AsyncMock(
            return_value=AsyncMock(
                candidates=[
                    MagicMock(
                        content=MagicMock(
                            parts=[MagicMock(text="Because it is filled with joy")]
                        )
                    )
                ]
            )
        )
        mock_second_response = AsyncMock(
            return_value=AsyncMock(
                candidates=[
                    MagicMock(
                        content=MagicMock(
                            parts=[
                                MagicMock(text="Because you don't understand colors")
                            ]
                        )
                    )
                ]
            )
        )

        # Assign the mock responses to the respective instances
        mock_get_client.return_value.generate_content_async.side_effect = [
            mock_first_response(),  # Response for first_chat
            mock_second_response(),  # Response for second_chat
        ]

        # Call mocked models
        first_response = await first_chat.get_response(message="Why is the sky green?")
        second_response = await second_chat.get_response(
            message="Why is the sky green?"
        )

        # Assert that the responses are as expected based on the system prompts
        assert "filled with joy" in first_response[0].lower()
        assert "don't understand colors" in second_response[0].lower()


@pytest.fixture
def vertex_chat() -> VertexAIChat:
    return VertexAIChat(
        model_id="gemini-1.5-pro",
        gcp_project_id="gcp-project",
        max_output_tokens=10,
        max_retries=2,
        initial_delay=0.1,
        exponential_base=1.5,
    )
