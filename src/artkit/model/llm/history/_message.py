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
Implementation of ``ChatMessage``.
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod

from pytools.api import appenddoc, inheritdoc
from pytools.expression import Expression, HasExpressionRepr
from pytools.expression.atomic import Id

log = logging.getLogger(__name__)

__all__ = [
    "AssistantMessage",
    "ChatMessage",
    "CustomMessage",
    "UserMessage",
    "SystemPrompt",
]


@inheritdoc(match="""[see superclass]""")
class ChatMessage(HasExpressionRepr, metaclass=ABCMeta):
    """
    A message that is part of a :class:`.ChatHistory`.
    """

    #: The content of the message.
    text: str

    def __init__(self, text: str) -> None:
        """
        :param text: the content of the message
        """
        self.text = text

    @property
    @abstractmethod
    def role(self) -> str:
        """
        The role of the speaker.
        """

    def to_expression(self) -> Expression:
        """[see superclass]"""
        return Id(type(self))(**vars(self))


@inheritdoc(match="""[see superclass]""")
class UserMessage(ChatMessage):
    """
    A chat message from a user.
    """

    #: The name of the user role, returned by :attr:`.role`.
    ROLE = "user"

    @property
    def role(self) -> str:
        """[see superclass]"""
        return self.ROLE


class AssistantMessage(ChatMessage):
    """
    A chat message from an assistant.
    """

    #: The name of the agent role, returned by :attr:`.role`.
    ROLE = "assistant"

    @property
    def role(self) -> str:
        """[see superclass]"""
        return self.ROLE


class SystemPrompt(ChatMessage):
    """
    A system prompt.
    """

    #: The name of the system role, returned by :attr:`.role`.
    ROLE = "system"

    @property
    def role(self) -> str:
        """[see superclass]"""
        return self.ROLE


class CustomMessage(ChatMessage):
    """
    A chat message with a custom role.
    """

    _role: str

    @appenddoc(to=ChatMessage.__init__)
    def __init__(self, text: str, *, role: str) -> None:
        """
        :param role: the role of the speaker
        """
        super().__init__(text=text)
        self._role = role

    @property
    def role(self) -> str:
        """[see superclass]"""
        return self._role
