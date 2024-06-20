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
Implementation of ``ChatFromCompletionModel``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Generic, TypeVar

from pytools.api import inheritdoc
from pytools.text import TextTemplate

from ..base import GenAIModelAdapter
from .base import ChatModel, CompletionModel
from .history import ChatHistory

log = logging.getLogger(__name__)

__all__ = [
    "ChatFromCompletionModel",
]

#
# Type variables
#

T_CompletionModel = TypeVar("T_CompletionModel", bound=CompletionModel)
T_ChatFromCompletionModel = TypeVar(
    "T_ChatFromCompletionModel", bound="ChatFromCompletionModel[Any]"
)
# use Any to prevent PyCharm code inspection error
_ = Any

#
# CONSTANTS
#


_DEFAULT_CHAT_TEMPLATE = """\
In the following conversation, a [USER] message is answered by an [AGENT].
{system_prompt}
[USER] {user_prompt} [/USER]
[AGENT]"""

#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class ChatFromCompletionModel(
    GenAIModelAdapter[T_CompletionModel], ChatModel, Generic[T_CompletionModel]
):
    """
    An adapter that turns a text completion generator into a chat system.

    The chat system generates responses to user prompts by using a chat template to
    format the user prompt and the system prompt into a single prompt for the text
    generator.

    The chat template must include the formatting keys ``system_prompt`` and
    ``user_prompt``.

    The default chat template is:

    .. code-block:: text

        In the following conversation, a [USER] message is answered by an [AGENT].
        {system_prompt}
        [USER] {user_prompt} [/USER]
        [AGENT]
    """

    #: The chat template to use.
    _chat_template: TextTemplate

    #: The system prompt to use.
    _system_prompt: str | None

    #: The default chat template.
    DEFAULT_CHAT_TEMPLATE: str = _DEFAULT_CHAT_TEMPLATE

    def __init__(
        self,
        model: T_CompletionModel,
        *,
        system_prompt: str | None = None,
        chat_template: str | None = None,
    ) -> None:
        """
        :param model: the text generator to use as the basis for the chat system
        :param system_prompt: the system prompt to use (optional)
        :param chat_template: the chat template to use (optional, defaults to
            :attr:`.DEFAULT_CHAT_TEMPLATE`)
        """
        super().__init__(model=model)
        self._system_prompt = system_prompt
        self._chat_template = TextTemplate(
            format_string=chat_template or self.DEFAULT_CHAT_TEMPLATE,
            required_keys=["system_prompt", "user_prompt"],
        )

    @property
    def system_prompt(self) -> str | None:
        """[see superclass]"""
        return self._system_prompt

    @property
    def chat_template(self) -> str:
        """
        The chat template to use.
        """
        return self._chat_template.format_string

    def with_system_prompt(
        self: T_ChatFromCompletionModel, system_prompt: str
    ) -> T_ChatFromCompletionModel:
        """[see superclass]"""
        new = copy.copy(self)
        new._system_prompt = system_prompt
        return new

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""
        if history is not None:
            raise ValueError("Chat history is not supported for completion models")
        return self.postprocess_response(
            await self.model.get_completion(
                prompt=self.preprocess_prompt(message), **model_params
            )
        )

    def preprocess_prompt(self, user_prompt: str) -> str:
        """
        Preprocess the user prompt before passing it to the text generator,
        substituting the system and user prompts into the chat template.

        :param user_prompt: the user prompt to preprocess
        :return: the prompt to be passed to the text generator
        """
        return self._chat_template.format_with_attributes(
            system_prompt=self.system_prompt or "", user_prompt=user_prompt
        )

    # noinspection PyMethodMayBeStatic
    def postprocess_response(self, response: str) -> list[str]:
        """
        Post-process the response.

        By default, strips leading and trailing whitespace, removes a trailing
        ``[/AGENT]`` tag and subsequent text if present, and returns the response as a
        single-item list.

        Can be overridden in subclasses.

        :param response: the response to postprocess
        :return: the post-processed response
        """
        agent_tag_position = response.rfind("[/AGENT]")
        if agent_tag_position >= 0:
            response = response[:agent_tag_position]
        return [response.strip()]
