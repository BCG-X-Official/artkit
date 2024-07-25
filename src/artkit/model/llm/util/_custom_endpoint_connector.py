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
Abstract base class to represent connecting to a custom endpoint.
"""
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from pytools.api import MissingClassMeta, appenddoc, inheritdoc, subsdoc

from ...util import RateLimitException
from ..base import ChatModelConnector
from ..history import ChatHistory

try:
    from aiohttp import ClientResponse, ClientResponseError, ClientSession

except ImportError:

    class ClientResponseError(metaclass=MissingClassMeta, module="aiohttp"):  # type: ignore
        """Placeholder class for missing ``ClientResponseError`` class."""

    class ClientSession(metaclass=MissingClassMeta, module="aiohttp"):  # type: ignore
        """Placeholder class for missing ``ClientSession`` class."""

    class ClientResponse(metaclass=MissingClassMeta, module="aiohttp"):  # type: ignore
        """Placeholder class for missing ``ClientResponse`` class."""


log = logging.getLogger(__name__)

__all__ = ["CustomChatEndpointConnector"]

#
# Type variables
#

T_CustomChatEndpointConnector = TypeVar(
    "T_CustomChatEndpointConnector", bound="CustomChatEndpointConnector"
)


@inheritdoc(match="""[see superclass]""")
class CustomChatEndpointConnector(ChatModelConnector[None], metaclass=ABCMeta):
    """
    Abstract base class to represent connecting to a custom endpoint.
    """

    url: str | None

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return ""

    @classmethod
    def _make_client(self) -> None:
        return None

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
        url: str | None = None,
        **model_params: Any,
    ) -> None:
        """
        :param url: the endpoint where the URL is.
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
        self.url = url

    @abstractmethod
    def format_message(self, message: str) -> str:
        """
        This method is responsible for formatting the input to the LLM chat system.

        :param message: The input message to format.
        :return: The formatted message.
        """

        pass

    @abstractmethod
    def format_headers(self) -> dict[str, Any]:
        """
        This method is responsible for formatting the headers to the request.

        :return: A dictionary of headers.
        """

        pass

    @abstractmethod
    async def format_response(self, response: ClientResponse) -> list[str]:
        """
        This method is responsible for formatting the response from the LLM chat.

        :param response: The response from the endpoint.
        :return: A list of formatted response strings.
        """
        pass

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""
        async with AsyncExitStack():
            async with ClientSession(headers=self.format_headers()) as aio_session:
                if self.url is None:
                    raise ValueError("URL must be provided")
                response = await aio_session.post(
                    url=self.url, data=self.format_message(message=message)
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
        return await self.format_response(response=response)
