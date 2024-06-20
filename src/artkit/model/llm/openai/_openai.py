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
Implementation of llm module.
"""

import logging
from collections.abc import Iterator
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from pytools.api import MissingClassMeta, inheritdoc

from ...util import RateLimitException
from ..base import ChatModelConnector
from ..history import ChatHistory

log = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, RateLimitError
    from openai.types.chat import ChatCompletion
except ImportError:  # pragma: no cover

    class AsyncOpenAI(metaclass=MissingClassMeta, module="openai"):  # type: ignore
        """Placeholder class for missing ``AsyncOpenAI`` class."""

    class ChatCompletion(  # type: ignore
        metaclass=MissingClassMeta, module="openai.types.chat"
    ):
        """Placeholder class for missing ``ChatCompletion`` class."""

    class RateLimitError(metaclass=MissingClassMeta, module="openai"):  # type: ignore
        """Placeholder class for missing ``RateLimitError`` class."""


__all__ = [
    "OpenAIChat",
]


#
# Type variables
#

T_OpenAIChat = TypeVar("T_OpenAIChat", bound="OpenAIChat")

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class OpenAIChat(ChatModelConnector[AsyncOpenAI]):
    """
    An asynchronous OpenAI LLM.
    """

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return "OPENAI_API_KEY"

    def _make_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.get_api_key())

    def _messages_to_openai_format(
        self, user_message: str, *, history: ChatHistory | None = None
    ) -> Iterator[dict[str, str]]:
        """
        Get the messages to send to the OpenAI LLM, based on the given user prompt
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
                log.warning(
                    "Expected only assistant messages, but got completion choice "
                    f"{choice!r}"
                )
            yield str(message.content)
