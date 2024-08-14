import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import boto3
import pytest
from httpx import HTTPStatusError, Response
from moto import mock_aws

from artkit.model.llm.bedrock import TitanBedrockChat
from artkit.model.util import RateLimitException

#######################################################################################
#                                     Constants                                       #
#######################################################################################
MODEL_ID = "amazon.titan-text-lite-v1"
REGION = "us-east-1"

MESSAGE = "What is the color of the sky?"
RESPONSE_TEXT = "The sky is blue."
PROMPT_COMPLETION = "blue"


@pytest.mark.asyncio
async def test_get_response(mock_bedrock_chat: TitanBedrockChat) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
    ) as MockClientSession:
        mock_request = Mock()
        mock_request.json = Mock(
            return_value={"results": [{"outputText": RESPONSE_TEXT}]}
        )
        mock_request.text = AsyncMock()
        mock_request.return_value.status_code = 200

        mock_connection = AsyncMock()
        mock_connection.request.return_value = mock_request
        MockClientSession.return_value = mock_connection

        response = await mock_bedrock_chat.get_response(message=MESSAGE)
        assert response == [RESPONSE_TEXT]


@pytest.mark.asyncio
async def test_rate_limit_error(mock_bedrock_chat: TitanBedrockChat) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
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
            await mock_bedrock_chat.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_invalid_request_error(mock_bedrock_chat: TitanBedrockChat) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
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
            await mock_bedrock_chat.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_unexpected_error(mock_bedrock_chat: TitanBedrockChat) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
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
            await mock_bedrock_chat.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_response_parsing(mock_bedrock_chat: TitanBedrockChat) -> None:
    response = Mock(spec=Response)
    response_body = {"results": [{"outputText": RESPONSE_TEXT}]}
    response.json = Mock(return_value=response_body)
    responses = mock_bedrock_chat.parse_httpx_response(response=response)
    assert responses == [RESPONSE_TEXT]


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################
@pytest.fixture(scope="function")
def mock_bedrock_chat(aws_credentials: Any) -> Generator[TitanBedrockChat, None, None]:
    with mock_aws():
        # Mock the AWS credentials
        boto3.client("sts").get_caller_identity()

        yield TitanBedrockChat(
            model_id=MODEL_ID,
            region=REGION,
            initial_delay=0.1,
            exponential_base=1.5,
            max_retries=2,
        )


@pytest.fixture(scope="function")
def aws_credentials() -> None:
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
