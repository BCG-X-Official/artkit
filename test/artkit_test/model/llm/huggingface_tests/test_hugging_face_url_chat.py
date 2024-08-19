from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import HTTPStatusError, Response

from artkit.model.llm.base import ChatModel
from artkit.model.llm.huggingface import HuggingfaceURLChat
from artkit.model.util import RateLimitException

_ = pytest.importorskip("huggingface_hub")

#######################################################################################
#                                     Constants                                       #
#######################################################################################

EXAMPLE_URL = "http://huggingface.com"

#######################################################################################
#                                     UTILs                                           #
#######################################################################################


@pytest.mark.asyncio
async def test_huggingface_url_chat(
    hugging_face_url_chat: HuggingfaceURLChat,
) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
    ) as MockClientSession:

        # Mock the response object
        mock_post = Mock()
        mock_post.json = Mock(
            return_value={
                "choices": [{"message": {"role": "assistant", "content": "blue"}}]
            }
        )
        mock_post.text = AsyncMock()
        mock_post.return_value.status = 200

        # Set up the mock connection object
        mock_connection = AsyncMock()
        mock_connection.request.return_value = mock_post
        MockClientSession.return_value = mock_connection

        assert (
            await hugging_face_url_chat.with_system_prompt(
                "Your job is to answer a quiz question with a single word, "
                "lowercase, with no punctuation"
            ).get_response(message="What color is the sky?")
        )[0] == "blue"


@pytest.mark.asyncio
async def test_retry_huggingface_url_chat(
    hugging_face_url_chat: HuggingfaceURLChat,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
    ) as MockClientSession:

        # Set up the mock connection object
        mock_connection = AsyncMock()

        def f() -> None:
            err = HTTPStatusError(
                request=Mock(),
                response=Mock(spec=Response, status_code=429),
                message="Rate limit exceeded",
            )
            raise err

        mock_connection.request.return_value.raise_for_status = f
        MockClientSession.return_value = mock_connection

        # Test that request is being retried
        n_retries = hugging_face_url_chat.max_retries
        with pytest.raises(RateLimitException):
            assert (
                await hugging_face_url_chat.with_system_prompt(
                    "Your job is to answer a quiz question with a single word, "
                    "lowercase, with no punctuation"
                ).get_response(message="What color is the sky?")
            )[0] == "blue"
        assert mock_connection.request.call_count == n_retries

    # Test that warnings are given
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == n_retries
    )


@pytest.mark.asyncio
async def test_unprocessable_huggingface_chat_aiohttp(
    hugging_face_url_chat: HuggingfaceURLChat,
) -> None:
    with patch(
        "artkit.model.llm.base.HTTPXChatConnector.get_client"
    ) as MockClientSession:

        # Set up the mock connection object
        mock_connection = AsyncMock()
        mock_connection.request.return_value.text.return_value = "Too many tokens"

        def f() -> None:
            err = HTTPStatusError(
                request=Mock(),
                response=Mock(spec=Response, status_code=422),
                message="Too many tokens",
            )
            raise err

        mock_connection.request.return_value.raise_for_status = f
        MockClientSession.return_value = mock_connection

        # Test that request fails
        with pytest.raises(ValueError):
            await hugging_face_url_chat.get_response(message="What color is the sky?")


def test_for_system_prompt(hugging_face_url_chat: HuggingfaceURLChat) -> None:
    chat = hugging_face_url_chat.with_system_prompt(
        "Your job is to answer a quiz question with a single word, "
        "lowercase, with no punctuation"
    )
    assert chat.system_prompt == (
        "Your job is to answer a quiz question with a single word, "
        "lowercase, with no punctuation"
    )


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################


@pytest.fixture
def hugging_face_url_chat(hf_token: str) -> ChatModel:
    return HuggingfaceURLChat(
        max_new_tokens=1,
        temperature=1.0,
        api_key_env=hf_token,
        model_id=EXAMPLE_URL,
        initial_delay=0.1,
        exponential_base=1.5,
        max_retries=2,
    )
