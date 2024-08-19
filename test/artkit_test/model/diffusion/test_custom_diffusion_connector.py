import base64
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import AsyncClient, HTTPStatusError, Response

from artkit.model.diffusion.base import HTTPXDiffusionConnector
from artkit.model.util import RateLimitException
from artkit.util import Image

#######################################################################################
#                                     Constants                                       #
#######################################################################################
MODEL_ID = "test_model"
URL = "http://test.url"

MESSAGE = "A cat with wings"
IMAGE_DATA = b"dGVzdA=="  # Base64 for "test"
IMAGE = Image(data=IMAGE_DATA)


@pytest.mark.asyncio
async def test_text_to_image(
    mock_custom_connector: HTTPXDiffusionConnector,
) -> None:
    with patch(
        "artkit.model.diffusion.base.HTTPXDiffusionConnector.get_client"
    ) as MockClientSession:
        mock_post = Mock()
        mock_post.json = Mock(return_value={"images": [IMAGE_DATA.decode()]})
        mock_post.text = Mock()
        mock_post.return_value.status_code = 200

        mock_connection = AsyncMock(spec=AsyncClient)
        mock_connection.request.return_value = mock_post
        MockClientSession.return_value = mock_connection

        response = await mock_custom_connector.text_to_image(text=MESSAGE)
        assert response[0].data == base64.b64decode(IMAGE_DATA)


@pytest.mark.asyncio
async def test_rate_limit_error(
    mock_custom_connector: HTTPXDiffusionConnector,
) -> None:
    with patch(
        "artkit.model.diffusion.base.HTTPXDiffusionConnector.get_client"
    ) as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_rate_limit_error() -> None:
            err = HTTPStatusError(
                request=Mock(),
                response=Mock(spec=Response, status_code=429),
                message="Rate limit exceeded",
            )
            raise err

        mock_connection.request.return_value.raise_for_status = raise_rate_limit_error
        MockClientSession.return_value = mock_connection

        with pytest.raises(RateLimitException):
            await mock_custom_connector.text_to_image(text=MESSAGE)


@pytest.mark.asyncio
async def test_invalid_request_error(
    mock_custom_connector: HTTPXDiffusionConnector,
) -> None:
    with patch(
        "artkit.model.diffusion.base.HTTPXDiffusionConnector.get_client"
    ) as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_invalid_request_error() -> None:
            err = HTTPStatusError(
                request=Mock(),
                response=Mock(spec=Response, status_code=422),
                message="Invalid request",
            )
            raise err

        mock_connection.request.return_value.raise_for_status = (
            raise_invalid_request_error
        )
        MockClientSession.return_value = mock_connection

        with pytest.raises(ValueError):
            await mock_custom_connector.text_to_image(text=MESSAGE)


@pytest.mark.asyncio
async def test_unexpected_error(
    mock_custom_connector: HTTPXDiffusionConnector,
) -> None:
    with patch(
        "artkit.model.diffusion.base.HTTPXDiffusionConnector.get_client"
    ) as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_unexpected_error() -> None:
            err = HTTPStatusError(
                request=Mock(),
                response=Mock(spec=Response, status_code=500),
                message="Internal server error",
            )
            raise err

        mock_connection.request.return_value.raise_for_status = raise_unexpected_error
        MockClientSession.return_value = mock_connection

        with pytest.raises(HTTPStatusError):
            await mock_custom_connector.text_to_image(text=MESSAGE)


@pytest.mark.asyncio
async def test_response_parsing(
    mock_custom_connector: HTTPXDiffusionConnector,
) -> None:
    mock_response = AsyncMock()
    mock_response.json = Mock(return_value={"images": [IMAGE_DATA.decode()]})

    responses = mock_custom_connector.parse_httpx_response(mock_response)
    assert responses == [Image(data=base64.b64decode(IMAGE_DATA))]


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################
@pytest.fixture(scope="function")
def mock_custom_connector() -> Generator[HTTPXDiffusionConnector, None, None]:
    class MockCustomDiffusionEndpointConnector(HTTPXDiffusionConnector):
        def build_request_arguments(
            self, text: str, **model_params: dict[str, Any]
        ) -> dict[str, Any]:
            return dict(
                method="POST",
                url=self.model_id,
                messsage=f"Formatted: {text}",
                headers={"Authorization": "Bearer test_token"},
            )

        def parse_httpx_response(self, response: Response) -> list[Image]:
            json_response = response.json()
            return [
                Image(data=base64.b64decode(image.encode()))
                for image in json_response["images"]
            ]

    api_key_env = "TEST"
    os.environ[api_key_env] = "test"

    yield MockCustomDiffusionEndpointConnector(
        model_id=MODEL_ID,
        api_key_env=api_key_env,
        max_retries=2,
        initial_delay=0.1,
        exponential_base=1.5,
        url=URL,
    )
