import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import ClientResponseError
from huggingface_hub import AsyncInferenceClient
from transformers import PreTrainedTokenizerBase

from artkit.model.llm import (
    CachedChatModel,
    CachedCompletionModel,
    ChatFromCompletionModel,
)
from artkit.model.llm.base import ChatModel, ChatModelConnector, CompletionModel
from artkit.model.llm.huggingface import HuggingfaceChat, HuggingfaceCompletion
from artkit.model.util import RateLimitException

_ = pytest.importorskip("huggingface_hub")

#######################################################################################
#                                     Constants                                       #
#######################################################################################

MODEL_MISTRAL_8x7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
MODEL_GPT2 = "openai-community/gpt2"
ZEPHYR_7B_MODEL = "HuggingFaceH4/zephyr-7b-beta"
EXAMPLE_URL = "http://huggingface.com"

PROMPT = "In one and only one word, the color of the sky is:"
PROMPT_COMPLETION = "blue"

#######################################################################################
#                                     UTILs                                           #
#######################################################################################


def _identity(x: Any, *_args: Any, **_kwargs: Any) -> Any:
    return x


async def _async_identity(x: Any, *_args: Any, **_kwargs: Any) -> Any:
    return x


@pytest.mark.asyncio
async def test_huggingface_chat(
    cached_huggingface_completion_chat: ChatFromCompletionModel[Any],
) -> None:
    """
    Test the HuggingfaceLLM class.
    """
    assert (
        await cached_huggingface_completion_chat.with_system_prompt(
            "Your job is to answer a quiz question in a single word"
        ).get_response(message="What color is the sky?")
    )[0].strip().lower() == PROMPT_COMPLETION

    # check that [/AGENT] gets clipped
    assert cached_huggingface_completion_chat.postprocess_response("test [/AGENT]") == [
        "test"
    ]


@pytest.mark.asyncio
async def test_huggingface_async(cached_huggingface_async: CompletionModel) -> None:
    """
    Test the AsyncHuggingfaceLLM class.
    """
    assert (
        await cached_huggingface_async.get_completion(prompt=PROMPT)
    ).strip().lower() == PROMPT_COMPLETION


@pytest.mark.asyncio
async def test_cached_huggingface_chat_async(
    cached_huggingface_chat_async: ChatModel,
) -> None:
    """
    Test the HuggingfaceChat class.
    """
    assert (
        await cached_huggingface_chat_async.with_system_prompt(
            "Your job is to answer a quiz question with a single word, "
            "lowercase, with no punctuation"
        ).get_response(message="What color is the sky?")
    )[0] == "blue"


@pytest.mark.asyncio
async def test_huggingface_chat_async(
    hugging_face_chat: ChatModelConnector[AsyncInferenceClient],
    zephyr_7b_tokenizer: PreTrainedTokenizerBase,
) -> None:
    with patch("aiohttp.ClientSession") as MockClientSession:

        # Mock the response object
        mock_post = Mock()
        mock_post.read = AsyncMock(return_value=b'[{"generated_text": "blue"}]')
        mock_post.return_value.status = 200

        # Set up the mock connection object
        mock_connection = AsyncMock()
        mock_connection.post.return_value = mock_post
        MockClientSession.return_value = mock_connection

        with patch(
            "transformers.AutoTokenizer.from_pretrained"
        ) as mock_from_pretrained:
            mock_from_pretrained.return_value = zephyr_7b_tokenizer
            assert (
                await hugging_face_chat.with_system_prompt(
                    "Your job is to answer a quiz question with a single word, "
                    "lowercase, with no punctuation"
                ).get_response(message="What color is the sky?")
            )[0] == "blue"


@pytest.mark.asyncio
async def test_huggingface_retry(
    hugging_face_chat: HuggingfaceChat,
    zephyr_7b_tokenizer: PreTrainedTokenizerBase,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch("aiohttp.ClientSession") as MockClientSession:

        # Mock the response object
        mock_post = AsyncMock()
        mock_post.read.return_value = b'[{"generated_text": "blue"}]'
        mock_post.json.return_value = {"error": "Rate limit exceeded"}
        mock_post.return_value.status = 429

        # Request info mock
        request_info = AsyncMock()
        request_info.real_url = (
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        )

        def f() -> None:
            err = ClientResponseError(
                request_info=request_info,
                history=AsyncMock(),
                status=429,
                message="Rate limit exceeded",
            )
            raise err

        mock_post.raise_for_status = f

        # Set up the mock connection object
        mock_connection = AsyncMock()
        mock_connection.post.return_value = mock_post
        MockClientSession.return_value = mock_connection

        # Get number of awaited retries
        n_retries = hugging_face_chat.max_retries

        with patch(
            "transformers.AutoTokenizer.from_pretrained"
        ) as mock_from_pretrained:
            mock_from_pretrained.return_value = zephyr_7b_tokenizer
            with pytest.raises(RateLimitException):
                await hugging_face_chat.with_system_prompt(
                    "Your job is to answer a quiz question with a single word, "
                    "lowercase, with no punctuation"
                ).get_response(message="What color is the sky?")
            assert mock_connection.post.call_count == n_retries

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


def test_error_on_missing_chat_template(
    zephyr_7b_tokenizer: PreTrainedTokenizerBase, hf_token: str
) -> None:
    with pytest.raises(ValueError):
        with patch(
            "transformers.AutoTokenizer.from_pretrained"
        ) as mock_from_pretrained:
            zephyr_7b_tokenizer.chat_template = None
            mock_from_pretrained.return_value = zephyr_7b_tokenizer
            HuggingfaceChat(
                max_new_tokens=1,
                temperature=1.0,
                api_key_env=hf_token,
                model_id=ZEPHYR_7B_MODEL,
            ).get_tokenizer()


@pytest.mark.asyncio
async def test_custom_chat_template(
    zephyr_7b_tokenizer: PreTrainedTokenizerBase, hf_token: str
) -> None:
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.return_value = zephyr_7b_tokenizer
        assert zephyr_7b_tokenizer.chat_template is not None
        llm = HuggingfaceChat(
            max_tokens=1,
            temperature=1.0,
            chat_template=(
                "{{ bos_token }}"
                "{% for message in messages %}"
                "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
                "{{ raise_exception("
                "'Conversation roles must alternate user/assistant/user/assistant/...'"
                ") }}"
                "{% endif %}"
                "{% if message['role'] == 'user' %}"
                "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token}}"
                "{% else %}"
                "{{ raise_exception('Only user and assistant roles are supported!') }}"
                "{% endif %}"
                "{% endfor %}"
            ),
            chat_template_params={"bos_token": "<s>"},
            api_key_env=hf_token,
            model_id=ZEPHYR_7B_MODEL,
        )

        mock_client = AsyncMock()
        mock_client.text_generation = _async_identity
        with patch.object(llm, "get_client") as mock_get_client:
            mock_get_client.return_value = mock_client
            assert (
                await llm._get_client_response([{"role": "user", "content": PROMPT}])
            )[0] == f"<s>[INST] {PROMPT} [/INST]"


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################


@pytest.fixture
def hf_token() -> str:
    api_key = "HF_TOKEN"
    # create a dummy key if not found
    if api_key not in os.environ:
        os.environ[api_key] = "<HF API key>"
    return api_key


@pytest.fixture
def cached_huggingface_async(
    hf_token: str, data_path: Path
) -> Iterator[CompletionModel]:
    database = data_path / "_copy_huggingface_async.db"
    shutil.copyfile(data_path / "huggingface_async.db", database)

    llm = HuggingfaceCompletion(
        api_key_env=hf_token, model_id=MODEL_MISTRAL_8x7B, max_tokens=1, temperature=1.0
    )

    yield CachedCompletionModel(model=llm, database=database)

    os.remove(database)


@pytest.fixture
def hugging_face_chat(
    zephyr_7b_tokenizer: PreTrainedTokenizerBase, hf_token: str
) -> ChatModel:

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.return_value = zephyr_7b_tokenizer

        return HuggingfaceChat(
            max_new_tokens=1,
            temperature=1.0,
            api_key_env=hf_token,
            model_id=ZEPHYR_7B_MODEL,
            initial_delay=0.1,
            exponential_base=1.5,
            max_retries=2,
        )


@pytest.fixture
def cached_huggingface_chat_async(
    hugging_face_chat: ChatModel, data_path: Path
) -> Iterator[ChatModel]:
    database = data_path / "_copy_huggingface_chat_async.db"
    shutil.copyfile(data_path / "huggingface_chat_async.db", database)

    yield CachedChatModel(model=hugging_face_chat, database=database)

    os.remove(database)


@pytest.fixture
def cached_huggingface_completion_chat(
    cached_huggingface_async: CompletionModel,
) -> ChatFromCompletionModel[Any]:
    return ChatFromCompletionModel(model=cached_huggingface_async)


@pytest.fixture
def zephyr_7b_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = PreTrainedTokenizerBase(clean_up_tokenization_spaces=True)
    tokenizer.chat_template = (
        "{% for message in messages %}\n{% if message['role'] == 'user' %}\n"
        "{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n"
        "{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n"
        "{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    )
    tokenizer.eos_token = "</s>"
    return tokenizer
