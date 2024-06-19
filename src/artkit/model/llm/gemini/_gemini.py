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
Implementation of the Gemini llm module.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from pytools.api import MissingClassMeta, appenddoc, inheritdoc, subsdoc

from ...util import RateLimitException
from ..base import ChatModelConnector
from ..history import ChatHistory

log = logging.getLogger(__name__)

try:  # pragma: no covers
    from google.api_core.exceptions import TooManyRequests
    from google.generativeai import GenerationConfig, GenerativeModel, configure
    from google.generativeai.types import AsyncGenerateContentResponse

except ImportError:

    class GenerativeModel(  # type: ignore[no-redef]
        metaclass=MissingClassMeta, module="google.generativeai"
    ):
        """Placeholder class for missing ``GenerativeModel`` class."""

    class GenerationConfig(  # type: ignore[no-redef]
        metaclass=MissingClassMeta, module="google.generativeai"
    ):
        """Placeholder class for missing ``GenerationConfig`` class."""

    class AsyncGenerateContentResponse(  # type: ignore[no-redef]
        metaclass=MissingClassMeta, module="google.generativeai.types"
    ):
        """Placeholder class for missing ``AsyncGenerateContentResponse`` class."""

    # noinspection PyPep8Naming
    class configure(  # type: ignore[no-redef]
        # pragma: no cover
        metaclass=MissingClassMeta,
        module="google.generativeai",
    ):
        """Placeholder class for missing ``configure`` function."""


__all__ = [
    "GeminiChat",
]


#
# Type variables
#

T_GeminiLLM = TypeVar("T_GeminiLLM", bound="GeminiChat")

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class GeminiChat(ChatModelConnector[GenerativeModel]):
    """
    An asynchronous Gemini LLM.
    """

    #: If ``False``, disable all accessible safety categories; otherwise, enable them.
    safety: bool

    @subsdoc(
        # The pattern matches the row defining model_params, and move it to the end
        # of the docstring.
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
        safety: bool = True,
        **model_params: Any,
    ) -> None:
        """
        :param safety: if ``False``, disable all accessible safety categories;
            otherwise, enable them (default: ``True``)
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
        self.safety = safety

    @classmethod
    def get_default_api_key_env(cls) -> str:  # pragma: no cover
        """[see superclass]"""
        return "GEMINI_API_KEY"

    def get_model_params(self) -> Mapping[str, Any]:
        """[see superclass]"""
        return dict(**super().get_model_params(), safety=self.safety)

    def _make_client(self) -> GenerativeModel:  # pragma: no cover

        configure(api_key=self.get_api_key())
        return GenerativeModel(self.model_id)

    def _get_safety_settings(self) -> dict[str, str] | None:
        """
        Get safety settings for Gemini.

        :return: a list of dicts describing safety settings by harm category
        """

        if self.safety:
            return None
        else:
            return {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }

    @staticmethod
    def _messages_to_gemini_format(
        user_message: str, *, history: ChatHistory | None = None
    ) -> Iterator[dict[str, Any]]:
        """
        Get the messages to send to the Gemini LLM, based on the given user prompt
        and chat history, and the system prompt for this LLM.

        :param user_message: the user prompt to send to the Gemini LLM
        :param history: the chat history to include in the messages (optional)
        :return: the messages object, in the format expected by the Gemini API
        """
        if history is not None:
            for message in history.messages:
                yield {"role": message.role, "parts": [message.text]}

        yield {"role": "user", "parts": [user_message]}

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""
        async with AsyncExitStack():

            config = GenerationConfig(**{**self.model_params, **model_params})
            try:
                response = await self.get_client().generate_content_async(
                    list(self._messages_to_gemini_format(message, history=history)),
                    generation_config=config,
                    safety_settings=self._get_safety_settings(),
                )
            except TooManyRequests as e:
                raise RateLimitException(
                    "Rate limit exceeded. Please try again later."
                ) from e

        return list(self._text_from_response(response))

    @staticmethod
    def _text_from_response(
        response: AsyncGenerateContentResponse,
    ) -> Iterator[str]:
        """
        Get the text from the given response.

        :param response: the response to process
        :return: the text of the responses
        """
        for candidate in response.candidates:
            yield candidate.content.parts[0].text
