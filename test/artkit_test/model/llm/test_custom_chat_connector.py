import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import HTTPStatusError, Response

from artkit.model.llm.base import HTTPXChatConnector
from artkit.model.llm.history import ChatHistory
from artkit.model.util import RateLimitException

#######################################################################################
#                                     Constants                                       #
#######################################################################################
MODEL_ID = "test_model"
URL = "http://test.url"

MESSAGE = "What is the color of the sky?"
RESPONSE_TEXT = "The sky is blue."


@pytest.mark.asyncio
async def test_get_response(mock_custom_connector: HTTPXChatConnector) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
    ) as MockClientSession:
        mock_post = Mock()
        mock_post.json = Mock(return_value={"results": [{"outputText": RESPONSE_TEXT}]})
        mock_post.text = AsyncMock()
        mock_post.return_value.status_code = 200

        mock_connection = AsyncMock()
        mock_connection.request.return_value = mock_post
        MockClientSession.return_value = mock_connection

        response = await mock_custom_connector.get_response(message=MESSAGE)
        assert response == [RESPONSE_TEXT]


@pytest.mark.asyncio
async def test_rate_limit_error(
    mock_custom_connector: HTTPXChatConnector,
) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
    ) as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_rate_limit_error() -> None:
            request = Mock()
            response = Response(
                status_code=429,
                content=b"Rate limit exceeded",
                request=request,
            )
            err = HTTPStatusError(
                response=response,
                message="Rate limit exceeded",
                request=request,
            )
            raise err

        mock_connection.request.return_value.raise_for_status = raise_rate_limit_error
        MockClientSession.return_value = mock_connection

        with pytest.raises(RateLimitException):
            await mock_custom_connector.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_invalid_request_error(
    mock_custom_connector: HTTPXChatConnector,
) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
    ) as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_invalid_request_error() -> None:
            request = Mock()
            response = Response(
                status_code=422,
                content=b"Internal server error",
                request=request,
            )
            err = HTTPStatusError(
                response=response,
                message="Invalid request",
                request=request,
            )
            raise err

        mock_connection.request.return_value.raise_for_status = (
            raise_invalid_request_error
        )
        MockClientSession.return_value = mock_connection

        with pytest.raises(ValueError):
            await mock_custom_connector.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_unexpected_error(
    mock_custom_connector: HTTPXChatConnector,
) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
    ) as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_unexpected_error() -> None:
            request = Mock()
            response = Response(
                status_code=500,
                content=b"Internal server error",
                request=request,
            )
            err = HTTPStatusError(
                response=response,
                message="Internal server error",
                request=request,
            )
            raise err

        mock_connection.request.return_value.raise_for_status = raise_unexpected_error
        MockClientSession.return_value = mock_connection

        with pytest.raises(HTTPStatusError):
            await mock_custom_connector.get_response(message=MESSAGE)


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################
@pytest.fixture(scope="function")
def mock_custom_connector() -> Generator[HTTPXChatConnector, None, None]:
    class MockCustomChatEndpointConnector(HTTPXChatConnector):
        @classmethod
        def get_default_api_key_env(cls) -> str:
            """[see superclass]"""
            return "TEST"

        def build_request_arguments(
            self,
            message: str,
            *,
            history: ChatHistory | None = None,
            **model_params: dict[str, Any],
        ) -> dict[str, Any]:
            return dict(
                method="POST",
                url=self.model_id,
                data={"message": f"Formatted: {message}"},
                headers={"Authorization": "Bearer test_token"},
            )

        def parse_httpx_response(self, response: Response) -> list[str]:
            json_response = response.json()
            return [result["outputText"] for result in json_response["results"]]

    api_key_env = "TEST"
    os.environ[api_key_env] = "test"

    yield MockCustomChatEndpointConnector(
        model_id=URL,
        api_key_env=api_key_env,
        initial_delay=0.1,
        exponential_base=1.5,
        max_retries=2,
    )
