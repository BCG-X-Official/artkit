import os
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

        # Create a dummy API key in the environment
        os.environ["GCP_PROJECT_ID"] = "my_gcp_project"

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


@pytest.fixture
def vertex_chat() -> VertexAIChat:
    return VertexAIChat(
        model_id="gemini-1.5-pro",
        gcp_project_id_env="GCP_PROJECT_ID",
        max_output_tokens=10,
        max_retries=2,
        initial_delay=0.1,
        exponential_base=1.5,
    )
