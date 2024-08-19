import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from artkit.model.llm import CachedCompletionModel
from artkit.model.llm.base import ChatModel, CompletionModel
from artkit.model.llm.huggingface import (
    HuggingfaceChatLocal,
    HuggingfaceCompletionLocal,
)

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


@pytest.mark.asyncio
async def test_huggingface_local(cached_huggingface_local: CompletionModel) -> None:
    """
    Test the HuggingfaceLocalLLM class.
    """

    assert cached_huggingface_local.model_id == MODEL_MISTRAL_7B

    assert (
        await cached_huggingface_local.get_completion(prompt=PROMPT)
    ).strip().lower() == PROMPT_COMPLETION


@pytest.mark.asyncio
async def test_local_hf_completion(
    huggingface_local_completion: CompletionModel,
) -> None:
    with patch(
        "artkit.model.llm.huggingface._huggingface_local.HuggingfaceCompletionLocal.get_client"
    ) as mock_get_client:
        with patch(
            "artkit.model.llm.huggingface._huggingface_local.HuggingfaceCompletionLocal.get_tokenizer"
        ) as mock_get_tokenizer:
            _RESULT = "blue"
            _MAP = {1: "What color", 2: " is the", 3: " sky?", 4: _RESULT}
            _PROMPT = "".join([_MAP[i] for i in range(1, 4)])

            # Setup mock client behavior
            mock_client = AsyncMock()
            mock_client.device = "cpu"
            mock_client.generate = lambda tokens, *_args, **_kwargs: torch.cat(
                (tokens, torch.Tensor([[4]])), dim=1
            )
            mock_get_client.return_value = mock_client

            # Setup mock tokenizer behavior
            mock_tokenizer = Mock()
            mock_tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])

            def mock_decode(tokens: torch.Tensor, *_args: Any, **_kwargs: Any) -> str:
                result = ""
                for token in tokens:
                    result += _MAP[token.item()]
                return result

            mock_tokenizer.decode = mock_decode
            mock_get_tokenizer.return_value = mock_tokenizer

            # Assert output being passed through correctly
            assert (
                await huggingface_local_completion.get_completion(prompt=_PROMPT)
            ) == _RESULT


@pytest.mark.asyncio
async def test_local_hf_chat(huggingface_local_chat: ChatModel) -> None:
    with patch(
        "artkit.model.llm.huggingface._huggingface_local.HuggingfaceChatLocal.get_client"
    ) as mock_get_client:
        with patch(
            "artkit.model.llm.huggingface._huggingface_local.HuggingfaceChatLocal.get_tokenizer"
        ) as mock_get_tokenizer:
            _PROMPT = "What color is the sky?"
            _RESULT = "blue"

            # Setup mock client behavior
            mock_client = AsyncMock()
            mock_client.device = "cpu"
            mock_client.generate = lambda tokens, *_args, **_kwargs: torch.cat(
                (tokens, torch.Tensor([[4]])), dim=1
            )
            mock_get_client.return_value = mock_client

            # Setup mock tokenizer behavior
            mock_tokenizer = Mock()
            mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])

            def mock_decode(tokens: torch.Tensor, *_args: Any, **_kwargs: Any) -> str:
                result = ""
                for token in tokens:
                    if token == 4:
                        result += _RESULT
                    else:
                        raise ValueError("Unexpected token")
                return result

            mock_tokenizer.decode = mock_decode
            mock_get_tokenizer.return_value = mock_tokenizer

            # Assert output being passed through correctly
            assert (await huggingface_local_chat.get_response(_PROMPT)) == [_RESULT]


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################


@pytest.fixture
def cached_huggingface_local(
    hf_token: str, data_path: Path
) -> Iterator[CompletionModel]:
    database = data_path / "_copy_huggingface_local.db"
    shutil.copyfile(data_path / "huggingface_local.db", database)

    yield CachedCompletionModel(
        model=HuggingfaceCompletionLocal(
            model_id=MODEL_MISTRAL_7B,
            api_key_env=hf_token,
            max_tokens=10,
            temperature=0.2,
        ),
        database=database,
    )

    os.remove(database)


@pytest.fixture
def huggingface_local_completion(hf_token: str) -> CompletionModel:
    return HuggingfaceCompletionLocal(
        api_key_env=hf_token,
        model_id=ZEPHYR_7B_MODEL,
        use_cuda=False,
    )


@pytest.fixture
def huggingface_local_chat(hf_token: str) -> ChatModel:
    return HuggingfaceChatLocal(
        api_key_env=hf_token,
        model_id=ZEPHYR_7B_MODEL,
        use_cuda=False,
    )


@pytest.fixture
def zephyr_7b_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = PreTrainedTokenizerBase()
    tokenizer.chat_template = (
        "{% for message in messages %}\n{% if message['role'] == 'user' %}\n"
        "{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n"
        "{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n"
        "{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    )
    tokenizer.eos_token = "</s>"
    return tokenizer
