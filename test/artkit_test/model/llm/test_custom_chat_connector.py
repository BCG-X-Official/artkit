import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import ClientResponse, ClientResponseError

from artkit.model.llm.util import CustomChatEndpointConnector
from artkit.model.util import RateLimitException

#######################################################################################
#                                     Constants                                       #
#######################################################################################
MODEL_ID = "test_model"
URL = "http://test.url"

MESSAGE = "What is the color of the sky?"
RESPONSE_TEXT = "The sky is blue."


@pytest.mark.asyncio
async def test_get_response(mock_custom_connector: CustomChatEndpointConnector) -> None:
    with patch("aiohttp.ClientSession.__aenter__") as MockClientSession:
        mock_post = Mock()
        mock_post.json = AsyncMock(
            return_value={"results": [{"outputText": RESPONSE_TEXT}]}
        )
        mock_post.text = AsyncMock()
        mock_post.return_value.status_code = 200

        mock_connection = AsyncMock()
        mock_connection.post.return_value = mock_post
        MockClientSession.return_value = mock_connection

        response = await mock_custom_connector.get_response(message=MESSAGE)
        assert response == [RESPONSE_TEXT]


@pytest.mark.asyncio
async def test_rate_limit_error(
    mock_custom_connector: CustomChatEndpointConnector,
) -> None:
    with patch("aiohttp.ClientSession.__aenter__") as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_rate_limit_error() -> None:
            err = ClientResponseError(
                request_info=AsyncMock(),
                history=AsyncMock(),
                status=429,
                message="Rate limit exceeded",
            )
            raise err

        mock_connection.post.return_value.raise_for_status = raise_rate_limit_error
        MockClientSession.return_value = mock_connection

        with pytest.raises(RateLimitException):
            await mock_custom_connector.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_invalid_request_error(
    mock_custom_connector: CustomChatEndpointConnector,
) -> None:
    with patch("aiohttp.ClientSession.__aenter__") as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_invalid_request_error() -> None:
            err = ClientResponseError(
                request_info=AsyncMock(),
                history=AsyncMock(),
                status=422,
                message="Invalid request",
            )
            raise err

        mock_connection.post.return_value.raise_for_status = raise_invalid_request_error
        MockClientSession.return_value = mock_connection

        with pytest.raises(ValueError):
            await mock_custom_connector.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_unexpected_error(
    mock_custom_connector: CustomChatEndpointConnector,
) -> None:
    with patch("aiohttp.ClientSession.__aenter__") as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_unexpected_error() -> None:
            err = ClientResponseError(
                request_info=AsyncMock(),
                history=AsyncMock(),
                status=500,
                message="Internal server error",
            )
            raise err

        mock_connection.post.return_value.raise_for_status = raise_unexpected_error
        MockClientSession.return_value = mock_connection

        with pytest.raises(ClientResponseError):
            await mock_custom_connector.get_response(message=MESSAGE)


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################
@pytest.fixture(scope="function")
def mock_custom_connector() -> Generator[CustomChatEndpointConnector, None, None]:
    class MockCustomChatEndpointConnector(CustomChatEndpointConnector):
        def format_message(self, message: str) -> str:
            return f"Formatted: {message}"

        def format_headers(self) -> dict[str, Any]:
            return {"Authorization": "Bearer test_token"}

        async def format_response(self, response: ClientResponse) -> list[str]:
            json_response = await response.json()
            return [result["outputText"] for result in json_response["results"]]

    api_key_env = "TEST"
    os.environ[api_key_env] = "test"

    yield MockCustomChatEndpointConnector(
        model_id=MODEL_ID,
        api_key_env=api_key_env,
        max_retries=2,
        url=URL,
    )
