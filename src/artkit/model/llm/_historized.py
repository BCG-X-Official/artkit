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
Implementation of ``HistorizedChatModel``.
"""

from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

from pytools.api import appenddoc

from .base import ChatModel, ChatModelAdapter
from .history import AssistantMessage, ChatHistory, UserMessage

log = logging.getLogger(__name__)

__all__ = [
    "HistorizedChatModel",
]


#
# Type variables
#

T_ChatModel_ret = TypeVar("T_ChatModel_ret", bound=ChatModel, covariant=True)


#
# Classes
#


class HistorizedChatModel(ChatModelAdapter[T_ChatModel_ret], Generic[T_ChatModel_ret]):
    """
    An LLM chat system that maintains a chat history.
    """

    #: The chat history.
    history: ChatHistory

    @appenddoc(to=ChatModelAdapter.__init__)
    def __init__(
        self,
        model: T_ChatModel_ret,
        *,
        max_history: int | None = None,
    ) -> None:
        """
        :param max_history: the maximum number of messages to store in the chat history
            (defaults to ``None`` for no limit)
        """
        super().__init__(model=model)
        self.history = ChatHistory(max_length=max_history)

    @property
    def max_history(self) -> int | None:
        """
        The maximum length of the chat history. ``None`` for no limit.
        """
        return self.history.max_length

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """
        Get a response, or multiple alternative responses, to a user
        message.

        Update the chat history with the user message and the first response.

        :param message: the user message
        :param history: must be ``None`` since the chat history is managed internally
        :param model_params: additional parameters for the chat system
        :return: the response or alternative responses
        :raises ValueError: if a history is passed as the ``history`` argument
        """
        if history is not None:
            raise ValueError("Cannot provide a history to a historized chat system")
        responses = await self.model.get_response(
            message=message, history=self.history, **model_params
        )
        self.history.add_message(UserMessage(message))
        if responses:
            self.history.add_message(AssistantMessage(responses[0]))
        return responses
