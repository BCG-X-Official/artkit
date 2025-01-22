import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError

from artkit.model.llm.azure import AzureOpenAIChat
from artkit.model.util import RateLimitException


@pytest.mark.asyncio
async def test_azureopenai(azureopenai_chat: AzureOpenAIChat) -> None:
    # Mock Azure openai Client
    with patch(
        "artkit.model.llm.azure._azureopenai.AsyncAzureOpenAI"
    ) as mock_get_client:
        # Mock Azure Openai Client response
        mock_response = AsyncMock(
            return_value=AsyncMock(
                choices=[MagicMock(message=MagicMock(content="blue", role="assistant"))]
            )
        )

        # Set mock response as return value
        mock_get_client.return_value.chat.completions.create = mock_response

        # Call mocked model
        messages = await azureopenai_chat.get_response(
            message="What color is the sky? Please answer in one word."
        )
        assert "blue" in messages[0].lower()


@pytest.mark.asyncio
async def test_azureopenai_retry(
    azureopenai_chat: AzureOpenAIChat, caplog: pytest.LogCaptureFixture
) -> None:
    # Mock openai Client
    with patch(
        "artkit.model.llm.azure._azureopenai.AsyncAzureOpenAI"
    ) as mock_get_client:
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
            await azureopenai_chat.get_response(
                message="What color is the sky? Please answer in one word."
            )
        assert (
            mock_get_client.return_value.chat.completions.create.call_count
            == azureopenai_chat.max_retries
        )
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == azureopenai_chat.max_retries
    )


@pytest.fixture
def azureopenai_chat() -> AzureOpenAIChat:
    env_vars = {
        "api_key": "AZURE_OPENAI_API_KEY",
        "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        "api_version": "OPENAI_API_VERSION",
        "deployment": "AZURE_OPENAI_DEPLOYMENT",
    }

    for env_var_key in env_vars.keys():
        os.environ[env_vars[env_var_key]] = "test"

    return AzureOpenAIChat(
        model_id="gpt-3.5-turbo",
        api_key_env=env_vars["api_key"],
        temperature=0.8,
        seed=0,
        max_output_tokens=10,
        max_retries=2,
        initial_delay=0.1,
        exponential_base=1.5,
    )
