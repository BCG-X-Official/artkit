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
from typing import Any

from pytools.api import MissingClassMeta, appenddoc, inheritdoc, subsdoc

from ...util import RateLimitException
from ..history import ChatHistory
from ._llm import ChatModelConnector

try:
    from httpx import AsyncClient, HTTPStatusError, Response

except ImportError:

    class AsyncClient(metaclass=MissingClassMeta, module="httpx"):  # type: ignore
        """Placeholder class for missing ``AsyncClient`` class."""

    class Response(metaclass=MissingClassMeta, module="httpx"):  # type: ignore
        """Placeholder class for missing ``Response`` class."""

    class HTTPStatusError(metaclass=MissingClassMeta, module="httpx"):  # type: ignore
        """Placeholder class for missing ``HTTPStatusError`` class."""


log = logging.getLogger(__name__)

__all__ = ["HTTPXChatConnector"]

#
# Type variables
#


@inheritdoc(match="""[see superclass]""")
class HTTPXChatConnector(ChatModelConnector[AsyncClient], metaclass=ABCMeta):
    """
    ABC that represents connecting to a custom endpoint.
    """

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
        initial_delay: float | None = None,
        exponential_base: float | None = None,
        jitter: bool | None = None,
        max_retries: int | None = None,
        system_prompt: str | None = None,
        httpx_client: AsyncClient | None = None,
        **model_params: Any,
    ) -> None:
        """
        :param httpx_client: the HTTPX client to use for making requests
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
        if httpx_client is None:
            httpx_client = AsyncClient()
        self.httpx_client = httpx_client

    def _make_client(self) -> AsyncClient:
        return self.httpx_client

    @abstractmethod
    def build_request_arguments(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        This method is responsible for formatting the input to the LLM chat system.
        For argument options see :class:`httpx.AsyncClient.request`.

        :param message: The input message to format.
        :param history: The chat history preceding the message.
        :param model_params: Additional parameters for the chat system.

        :return: The necessary httpx request arguments.
        """

    @abstractmethod
    def parse_httpx_response(self, response: Response) -> list[str]:
        """
        This method is responsible for formatting the :class:`httpx.Response` after
        having made the request.

        :param response: The response from the endpoint.
        :return: A list of formatted response strings.
        """

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""
        async with AsyncExitStack():
            response = await self.get_client().request(
                **self.build_request_arguments(
                    message=message, history=history, **model_params
                )
            )
            try:
                response.raise_for_status()
            except HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RateLimitException(
                        "Rate limit exceeded. Please try again later."
                    ) from e
                elif e.response.status_code == 422:
                    raise ValueError(
                        f"Invalid request. Please check the request parameters. {e.response.text}"
                    ) from e
                raise
        return self.parse_httpx_response(response=response)
