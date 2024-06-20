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
Implementation of the text producer.
"""

import logging
import re
from collections.abc import Iterable

from pytools.api import inheritdoc
from pytools.expression import (
    Expression,
    HasExpressionRepr,
    expression_from_init_params,
)
from pytools.text import TextTemplate

from .base import ChatModel

log = logging.getLogger(__name__)

__all__ = [
    "TextGenerator",
]


@inheritdoc(match="""[see superclass]""")
class TextGenerator(HasExpressionRepr):
    """
    Generates text using an LLM.

    Sets up an LLM with a system prompt that instructs the LLM to generate text items.

    The system prompt must:

    - include a set of required formatting keys, in the form of '{key}', that will be
      substituted with values before sending the system prompt to the LLM
    - instruct the LLM to accept an integer as the user prompt, and generate a number
      of texts indicated by the user prompt
    - instruct the LLM to separate generated text items by lines consisting of '#####'
    """

    #: The LLM system used to generate text items.
    llm: ChatModel

    def __init__(
        self, *, llm: ChatModel, system_prompt: str, formatting_keys: Iterable[str]
    ) -> None:
        """
        :param llm: the LLM system used to generate text
        :param system_prompt: the system prompt instructing the LLM to generate the
            text items
        :param formatting_keys: the names of the formatting keys used in the system
            prompt
        """
        super().__init__()
        self.llm = llm
        self._system_prompt_template = TextTemplate(
            format_string=system_prompt, required_keys=formatting_keys
        )

    @property
    def system_prompt(self) -> str:
        """
        The system prompt used to generate text items.
        """
        return self._system_prompt_template.format_string

    @property
    def formatting_keys(self) -> set[str]:
        """
        The formatting keys used in the system prompt.
        """
        return self._system_prompt_template.formatting_keys

    async def make_text_items(self, *, n: int, attributes: dict[str, str]) -> list[str]:
        """
        Use the LLM to generate the given number of text items, substituting the
        given attributes for the formatting keys in the system prompt.

        Calls method :meth:`parse_llm_response` to parse the LLM response into
        individual text items.

        :param n: the number of text items to generate
        :param attributes: the attributes to substitute for the formatting keys in
            the system prompt
        :return: the generated text items
        """

        # we make at least 10% more text items than requested, since the LLM sometimes
        # does not produce the full number of requested text items
        n_with_buffer = n * 11 // 10 + 1

        response = await self.llm.with_system_prompt(
            system_prompt=(
                self._system_prompt_template.format_with_attributes(**attributes)
            )
        ).get_response(message=str(n_with_buffer))

        # get the text items from the last response
        if len(response) > 1:
            log.warning("LLM returned more than one response; using the last one")
        text_items = self.parse_llm_response(response[-1])

        if len(text_items) < n:
            raise ValueError(
                f"Expected {n} text items but only got {len(text_items)} "
                f"in LLM response\n{response[-1]!r}"
            )

        return text_items[:n]

    @staticmethod
    def parse_llm_response(response: str) -> list[str]:
        """
        Parse an LLM response into individual text items.

        Unless overridden, splits the text using ``"#####"`` as the separator.
        Called by method :meth:`.make_text_items` with the LLM response.

        :param response: the LLM response to parse
        :return: a list of text items
        """
        return list(
            filter(
                # exclude empty descriptions, e.g., after a trailing "#####"
                None,
                map(
                    # strip whitespace
                    str.strip,
                    # split the response by the separator string
                    # be flexible with the number of '#' characters
                    # since the LLM may not use the precise number indicated
                    re.split(r"###+", response),
                ),
            )
        )

    def to_expression(self) -> Expression:
        """[see superclass]"""
        return expression_from_init_params(self)


#
# Auxiliary functions
#
