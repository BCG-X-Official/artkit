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
Implementation of ``MyClass``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

from pytools.api import as_list, inheritdoc
from pytools.expression import Expression, HasExpressionRepr
from pytools.expression.atomic import Id

from ._message import ChatMessage

log = logging.getLogger(__name__)

__all__ = [
    "ChatHistory",
]


#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class ChatHistory(HasExpressionRepr):
    """
    A chat history, which consists of a sequence of messages.

    Supports adding messages, getting the last ``n`` messages, iteration using
    :func:`iter` and getting the number of messages using :func:`len`.

    Chat messages can also be added using the ``+`` operator, which prepends or appends
    a message to the start or end of the chat history, respectively.

    ``history + message`` creates a new chat history with the message appended to the
    end, while ``message + history`` creates a new chat history with the message
    prepended to the start.
    """

    #: The messages in the chat history.
    messages: list[ChatMessage]

    #: The maximum length of the chat history, or ``None`` if there is no maximum
    #: length.
    max_length: int | None

    def __init__(
        self,
        messages: Iterable[ChatMessage] | None = None,
        *,
        max_length: int | None = None,
    ) -> None:
        """
        :param messages: the initial messages in the chat history (optional)
        :param max_length: the maximum length of the chat history
        """
        self.messages = (
            as_list(messages, element_type=ChatMessage, arg_name="messages")
            if messages is not None
            else []
        )
        if max_length is not None and max_length <= 0:
            raise ValueError(
                f"arg max_length must be positive or None, but got {max_length}"
            )
        self.max_length = max_length

    def add_message(self, message: ChatMessage) -> None:
        """
        Add a message to the chat history.

        :param message: the message to add
        """
        self.messages.append(message)
        if self.max_length is not None and len(self.messages) > self.max_length:
            del self.messages[0]

    def get_messages(self, n: int | None = None) -> list[ChatMessage]:
        """
        Get the last ``n`` messages in the chat history, or all messages if there are
        fewer than ``n`` messages or if ``n`` is ``None``.

        :param n: the number of messages to get (optional)
        :return: the last ``n`` messages
        """
        return self.messages if n is None else self.messages[-n:]

    def __iter__(self) -> Iterator[ChatMessage]:
        """
        Iterate over the messages in the chat history.

        :return: an iterator over the messages
        """
        return iter(self.messages)

    def __len__(self) -> int:
        """
        Get the number of messages in the chat history.

        :return: the number of messages
        """
        return len(self.messages)

    def to_expression(self) -> Expression:
        """[see superclass]"""
        return Id(type(self))(messages=self.messages)
