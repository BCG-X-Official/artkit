# -----------------------------------------------------------------------------
# © 2024 Boston Consulting Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------

"""
Huggingface LLM systems.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AsyncExitStack
from typing import Any, TypeVar, cast

from pytools.api import MissingClassMeta, inheritdoc

from ...util import RateLimitException
from ..base import ChatModelConnector
from ..history import ChatHistory, ChatMessage, SystemPrompt, UserMessage
from .base import (
    HuggingfaceChatConnectorMixin,
    HuggingfaceCompletionConnectorMixin,
    HuggingfaceRemoteConnectorMixin,
)

try:  # pragma: no cover
    from aiohttp import ClientResponseError, ClientSession
    from huggingface_hub import AsyncInferenceClient
except ImportError:

    class AsyncInferenceClient(  # type: ignore
        metaclass=MissingClassMeta, module="huggingface_hub"
    ):
        """Placeholder class for missing ``AsyncInferenceClient`` class."""

    class ClientResponseError(metaclass=MissingClassMeta, module="aiohttp"):  # type: ignore
        """Placeholder class for missing ``ClientResponseError`` class."""

    class ClientSession(metaclass=MissingClassMeta, module="aiohttp"):  # type: ignore
        """Placeholder class for missing ``ClientSession`` class."""


log = logging.getLogger(__name__)

__all__ = [
    "HuggingfaceCompletion",
    "HuggingfaceChat",
    "HuggingfaceURLChat",
]


#
# Type variables
#

T_Client = TypeVar("T_Client")

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class HuggingfaceCompletion(
    HuggingfaceRemoteConnectorMixin,
    HuggingfaceCompletionConnectorMixin[AsyncInferenceClient],
):
    """
    Asynchronous Huggingface remote API.
    """

    async def _make_completion(
        self, prompt: str, **model_params: dict[str, Any]
    ) -> str:  # pragma: no cover
        """[see superclass]"""

        try:
            return cast(
                str,
                await self.get_client().text_generation(
                    prompt,
                    **{**self.get_model_params(), **model_params},
                ),
            )
        except ClientResponseError as e:
            if e.status == 429:
                raise RateLimitException(
                    "Rate limit exceeded. Please try again later."
                ) from e
            raise e


@inheritdoc(match="""[see superclass]""")
class HuggingfaceChat(
    HuggingfaceRemoteConnectorMixin,
    HuggingfaceChatConnectorMixin[AsyncInferenceClient],
):
    """
    Asynchronous Huggingface API LLM.

    Examples:

    Model usage where chat template is implicitly derived from the tokenizer:

    .. code-block:: python

        import artkit.api as ak

        chat = ak.HuggingfaceChat(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            api_key_env="HF_TOKEN",
            template_kwargs={"add_generation_prompt": True}
        )

        response = await chat.get_response("Hello, how are you?")

    Set your own chat template:

    .. code-block:: python

        chat = ak.HuggingfaceChat(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            api_key_env="HF_TOKEN",
            custom_chat_template=(
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '' + message['role'] + '\\n\\n' + message['content'] | trim + '' %}"
                "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ 'assistant\\n\\n' }}"
                "{% endif %}"
            ),
            template_kwargs={"add_generation_prompt": True}
        )

        response = await chat.get_response("Hello, how are you?")
    """  # noqa: W505

    async def _get_client_response(
        self, messages: list[dict[str, str]], **model_params: dict[str, Any]
    ) -> list[str]:
        """[see superclass]"""

        messages = self.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            **self.chat_template_params,
        )
        try:
            response = await self.get_client().text_generation(
                messages,
                # return_full_text=False, stream=False, details=False,
                # ensure that the prompt is not included in the response
                # and that the response is a string
                **{
                    **{"return_full_text": False, "stream": False, "details": False},
                    # Can't use self.model_params() here since additional
                    # unrelated params are incorporated there
                    **self.model_params,
                    **model_params,
                },
            )
        except ClientResponseError as e:
            if e.status == 429:
                raise RateLimitException(
                    "Rate limit exceeded. Please try again later."
                ) from e
            raise
        if not isinstance(response, str):
            """
            stream=False, details=False, means the response should be a string
            we do not support other response types as of now as we currently
            only handle string responses for more information see:
            https://huggingface.co/docs/huggingface_hub/main/en/package_reference/ \
                inference_client#huggingface_hub.InferenceClient.text_generation
            """
            raise ValueError(f"Unexpected response type: {type(response)}")

        return [response.strip()]


@inheritdoc(match="""[see superclass]""")
class HuggingfaceURLChat(
    ChatModelConnector[None],
):
    """
    Asynchronous Huggingface API LLM using an URL as model_id.
    This class is used to interact with dedicated huggingface
    inference endpoints that use a URL as a model id.
    For regular hugging face models, use :class:`HuggingfaceChat`.

    .. note::

        This class uses the OpenAI API reference in the background
        to interact with the Huggingface API. Therefore, parameter
        names and behavior are defined here
        https://platform.openai.com/docs/api-reference/chat/create.
        This is done to avoid having to use a local tokenizer,
        while also wanting to pass a chat history.

    Examples:

    .. code-block:: python

        import artkit.api as ak

        chat = ak.HuggingfaceChat(
            model_id="http:/huggingface.com/v1/chat/completions",
        )

        response = await chat.get_response("Hello, how are you?")
    """

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return "HF_TOKEN"

    def _make_client(self) -> None:
        return None

    def _get_messages(
        self, message: str, *, history: ChatHistory | None
    ) -> Iterator[ChatMessage]:
        # Format the user prompt and system prompt (if defined)
        # as a list of chat messages for the Huggingface API.

        if self.system_prompt:
            yield SystemPrompt(self.system_prompt)

        if history is not None:
            yield from history.messages

        yield UserMessage(message)

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""
        async with AsyncExitStack():
            async with ClientSession(
                headers={
                    "Authorization": f"Bearer {self.get_api_key()}",
                    "Content-Type": "application/json",
                },
            ) as session:
                api_suffix = "/v1/chat/completions"
                url = (
                    self.model_id
                    if self.model_id.endswith(api_suffix)
                    else (self.model_id + api_suffix)
                )
                messages = [
                    {"role": message.role, "content": message.text}
                    for message in self._get_messages(message, history=history)
                ]
                response = await session.post(
                    url=url,
                    json={
                        "model": "tgi",
                        "messages": messages,
                        **{**self.model_params, **model_params},
                    },
                )
                response_text = await response.text()
                try:
                    # Raises exception if response status is not 200
                    response.raise_for_status()
                except ClientResponseError as e:
                    if e.status == 429:
                        raise RateLimitException(
                            "Rate limit exceeded. Please try again later."
                        ) from e
                    elif e.status == 422:
                        raise ValueError(
                            f"Invalid request. Please check the request parameters. {response_text}"
                        ) from e
                    raise

        response_body = await response.json()
        return list(self._responses_from_body(response_body))

    @staticmethod
    def _responses_from_body(
        response_body: dict[str, list[dict[str, dict[str, str]]]]
    ) -> Iterator[str]:
        """
        Get the response from the response_body.

        :param response_body: the chat response_body to process
        :return: the alternate responses from the chat response_body
        """

        for choice in response_body["choices"]:
            message = choice["message"]
            if message["role"] != "assistant":
                log.warning(
                    "Expected only assistant messages, but got completion choice "
                    f"{choice!r}"
                )
            yield str(message["content"])
