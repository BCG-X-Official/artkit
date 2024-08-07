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
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any, List, Optional, TypeVar

from pytools.api import MissingClassMeta, inheritdoc

from ...util import RateLimitException, retry_with_exponential_backoff
from ..base import ChatModel
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
class CustomChatEndpointConnector(ChatModel, metaclass=ABCMeta):
    """
    ABC that represents connecting to a custom endpoint.
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """
        :param model_id: the model_id or URL endpoint for the model to connect to.
        :param kwargs: additional keyword arguments passed to the constructor.
        """
        self._model_id = model_id
        self.initial_delay = kwargs.get("initial_delay", 1.0)
        self.exponential_base = kwargs.get("exponential_base", 2.0)
        self.jitter = kwargs.get("jitter", True)
        self.max_retries = kwargs.get("max_retries", 5)
        self.model_params = {k: v for k, v in kwargs.items() if v is not None}
        self._system_prompt = kwargs.get("system_prompt", None)

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

    @property
    def model_id(self) -> str:
        """[see superclass]"""
        return self._model_id

    @property
    def system_prompt(self) -> str | None:
        """[see superclass]"""
        return self._system_prompt

    def with_system_prompt(self, system_prompt: str) -> CustomChatEndpointConnector:
        """[see superclass]"""
        self._system_prompt = system_prompt
        return self

    def get_model_params(self) -> Mapping[str, Any]:
        """[see superclass]"""
        return self.model_params

    @retry_with_exponential_backoff
    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""
        async with AsyncExitStack():
            async with ClientSession() as session:
                response = await session.post(
                    self.model_id,
                    data=self.format_message(message),
                    headers=self.format_headers(),
                )
                response_text = await response.text()
                try:
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
