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
vLLM LLM systems.
"""
from __future__ import annotations

import logging
from abc import ABCMeta
from collections.abc import Iterator
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from artkit.model.llm.history._history import ChatHistory
from pytools.api import appenddoc, inheritdoc, subsdoc

from ...util import RateLimitException
from ..base import ChatModelConnector

logger = logging.getLogger(__name__)

__all__ = ["VLLMChat"]


T_VLLMChat = TypeVar("T_VLLMChat", bound="VLLMChat")


@inheritdoc(match="""[see superclass]""")
class VLLMChat(ChatModelConnector[AsyncOpenAI], metaclass=ABCMeta):
    """
    Base class for vLLM LLMs.
    """

    vllm_url: str

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """vLLM requires no API key since it's a self-managed server."""
        return ""

    def _make_client(self) -> AsyncOpenAI:  # pragma: no cover
        """
        This method handles the authentication and connection to the vLLM server.
        Since vLLM implements the OpenAI API spec, we can use the OpenAI client
        to connect to it.
        """
        return AsyncOpenAI(api_key="EMPTY", base_url=self.vllm_url)

    @subsdoc(
        pattern=r"(:param model_params: .*\n)((:?.|\n)*\S)(\n|\s)*",
        replacement=r"\2\1",
    )
    @appenddoc(to=ChatModelConnector.__init__)
    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str | None = None,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        system_prompt: str | None = None,
        vllm_url: str,
        **model_params: Any,
    ) -> None:
        """
        :param vllm_url: The URL of the vLLM server.
        """
        super().__init__(
            model_id=model_id,
            api_key_env=api_key_env,
            initial_delay=initial_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            max_retries=max_retries,
            system_prompt=system_prompt,
            **model_params,
        )
        self.vllm_url = vllm_url

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""
        async with AsyncExitStack():
            try:
                completion = await self.get_client().chat.completions.create(
                    messages=list(
                        self._messages_to_openai_format(  # type: ignore[arg-type]
                            message, history=history
                        )
                    ),
                    model=self.model_id,
                    **{**self.get_model_params(), **model_params},
                )
            except RateLimitError as e:
                raise RateLimitException(
                    "Rate limit exceeded. Please try again later."
                ) from e

        return list(self._responses_from_completion(completion))

    def _messages_to_openai_format(
        self, user_message: str, *, history: ChatHistory | None = None
    ) -> Iterator[dict[str, str]]:
        """
        Get the messages to send to the vLLM, based on the given user prompt
        and chat history, and the system prompt for this LLM.

        :param user_message: the user prompt to send to the OpenAI LLM
        :param history: the chat history to include in the messages (optional)
        :return: the messages object, in the format expected by the OpenAI API
        """
        if self.system_prompt:
            yield {"role": "system", "content": self.system_prompt}

        if history is not None:
            for message in history.messages:
                yield {"role": message.role, "content": message.text}

        yield {"role": "user", "content": user_message}

    @staticmethod
    def _responses_from_completion(completion: ChatCompletion) -> Iterator[str]:
        """
        Get the response from the given chat completion.

        :param completion: the chat completion to process
        :return: the alternate responses from the chat completion
        """

        for choice in completion.choices:
            message = choice.message
            if message.role != "assistant":
                logger.warning(
                    "Expected only assistant messages, but got completion choice "
                    f"{choice!r}"
                )
            yield str(message.content)
