import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import boto3
import pytest
from aiohttp import ClientResponseError
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
        "artkit.model.llm.bedrock.base._base.ClientSession.__aenter__"
    ) as MockClientSession:
        mock_post = Mock()
        mock_post.json = AsyncMock(
            return_value={"results": [{"outputText": RESPONSE_TEXT}]}
        )
        mock_post.text = AsyncMock()
        mock_post.return_value.status_code = 200

        mock_connection = AsyncMock()
        mock_connection.post.return_value = mock_post
        MockClientSession.return_value = mock_connection

        response = await mock_bedrock_chat.get_response(message=MESSAGE)
        assert response == [RESPONSE_TEXT]


@pytest.mark.asyncio
async def test_rate_limit_error(mock_bedrock_chat: TitanBedrockChat) -> None:
    with patch(
        "artkit.model.llm.bedrock.base._base.ClientSession.__aenter__"
    ) as MockClientSession:
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
            await mock_bedrock_chat.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_invalid_request_error(mock_bedrock_chat: TitanBedrockChat) -> None:
    with patch(
        "artkit.model.llm.bedrock.base._base.ClientSession.__aenter__"
    ) as MockClientSession:
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
            await mock_bedrock_chat.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_unexpected_error(mock_bedrock_chat: TitanBedrockChat) -> None:
    with patch(
        "artkit.model.llm.bedrock.base._base.ClientSession.__aenter__"
    ) as MockClientSession:
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
            await mock_bedrock_chat.get_response(message=MESSAGE)


@pytest.mark.asyncio
async def test_response_parsing(mock_bedrock_chat: TitanBedrockChat) -> None:
    response_body = {"results": [{"outputText": RESPONSE_TEXT}]}
    responses = mock_bedrock_chat._responses_from_body(response_body)
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
            max_retries=3,
        )


@pytest.fixture(scope="function")
def aws_credentials() -> None:
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
