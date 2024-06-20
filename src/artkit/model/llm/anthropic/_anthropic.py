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
Implementation of the Anthropic llm module.
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

try:  # pragma: no cover
    from anthropic import AsyncAnthropic, RateLimitError
    from anthropic.types import Message
except ImportError:

    class AsyncAnthropic(metaclass=MissingClassMeta, module="anthropic"):  # type: ignore
        """
        Placeholder class for missing ``AsyncAnthropic`` class.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Message(metaclass=MissingClassMeta, module="anthropic.types"):  # type: ignore
        """
        Placeholder class for missing ``Message`` class.
        """

    class RateLimitError(metaclass=MissingClassMeta, module="anthropic"):  # type: ignore
        """
        Placeholder class for missing ``RateLimitError`` class.
        """


__all__ = [
    "AnthropicChat",
]


#
# Type variables
#

T_AnthropicLLM = TypeVar("T_AnthropicLLM", bound="AnthropicChat")

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class AnthropicChat(ChatModelConnector[AsyncAnthropic]):
    """
    An asynchronous Anthropic LLM.
    """

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return "ANTHROPIC_API_KEY"

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
                message = await self.get_client().messages.create(
                    messages=list(
                        self._messages_to_anthropic_format(message, history=history)
                    ),
                    model=self.model_id,
                    **{
                        **(
                            {"system": self.system_prompt} if self.system_prompt else {}
                        ),
                        **self.get_model_params(),
                        **model_params,
                    },
                )
            except RateLimitError as e:
                raise RateLimitException(
                    "Rate limit exceeded. Please try again later."
                ) from e

        return list(self._responses_from_message(message))

    def _make_client(self) -> AsyncAnthropic:  # pragma: no cover
        return AsyncAnthropic(api_key=self.get_api_key())

    @staticmethod
    def _messages_to_anthropic_format(
        user_message: str, *, history: ChatHistory | None = None
    ) -> Iterator[dict[str, str]]:
        """
        Get the messages to send to the Anthropic LLM, based on the given user prompt
        and chat history, and the system prompt for this LLM.

        :param user_message: the user prompt to send to the Anthropic LLM
        :param history: the chat history to include in the messages (optional)
        :return: the messages object, in the format expected by the Anthropic API
        """
        if history is not None:
            for message in history.messages:
                yield {"role": message.role, "content": message.text}

        yield {"role": "user", "content": user_message}

    @staticmethod
    def _responses_from_message(message: Message) -> Iterator[str]:
        """
        Get the response from the given chat completion.

        :param message: the chat completion to process
        :return: the alternate responses from the chat completion
        """

        for content in message.content:
            if message.role != "assistant":
                log.warning("Expected only assistant messages, but got " f"{content!r}")
            yield content.text
