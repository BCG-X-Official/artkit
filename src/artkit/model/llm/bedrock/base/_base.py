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
Bedrock LLM systems.
"""
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from pytools.api import MissingClassMeta, appenddoc, inheritdoc, subsdoc

from ....util import RateLimitException
from ...base import ChatModelConnector
from ...history import ChatHistory

log = logging.getLogger(__name__)

__all__ = ["BaseBedrockChat"]

try:
    from aiohttp import ClientResponseError, ClientSession
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.session import Session

except ImportError:

    class ClientResponseError(metaclass=MissingClassMeta, module="aiohttp"):  # type: ignore
        """Placeholder class for missing ``ClientResponseError`` class."""

    class ClientSession(metaclass=MissingClassMeta, module="aiohttp"):  # type: ignore
        """Placeholder class for missing ``ClientSession`` class."""

    class SigV4Auth(metaclass=MissingClassMeta, module="SigV4Auth"):  # type: ignore
        """Placeholder class for missing ``SigV4Auth`` class."""

    class AWSRequest(metaclass=MissingClassMeta, module="AWSRequest"):  # type: ignore
        """Placeholder class for missing ``AWSRequest`` class."""

    class Session(metaclass=MissingClassMeta, module="Session"):  # type: ignore
        """Placeholder class for missing ``Session`` class."""


__all__ = ["BaseBedrockChat"]

#
# Type variables
#

T_BaseBedrockChat = TypeVar("T_BaseBedrockChat", bound="BaseBedrockChat")

#
# Class declarations
#
log = logging.getLogger(__name__)


@inheritdoc(match="""[see superclass]""")
class BaseBedrockChat(ChatModelConnector[None], metaclass=ABCMeta):
    """
    Base class for Bedrock LLMs.
    """

    region: str | None

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
        region: str | None = None,
        **model_params: Any,
    ) -> None:
        """
        :param region: The specific AWS region to connect to.
        :raises CredentialsNotFoundError: if unable to find AWS Credentials.
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

        self.region = region if region else "us-east-1"
        self.session = Session()
        self.credentials = self.session.get_credentials()
        self.auth = SigV4Auth(self.credentials, "bedrock", self.region)
        self.endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}/invoke"

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""

        request = AWSRequest(
            method="POST",
            url=self.endpoint,
            data=message,
            headers={"content-type": "application/json"},
        )
        self.auth.add_auth(request)
        prepped_request = request.prepare()
        headers = dict(prepped_request.headers.items())
        async with AsyncExitStack():
            async with ClientSession(headers=headers) as aio_session:
                response = await aio_session.post(
                    url=prepped_request.url, data=prepped_request.body
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

        response_body = await response.json()
        return list(self._responses_from_body(response_body))

    @abstractmethod
    def _responses_from_body(self, response_body: dict[str, Any]) -> list[str]:
        """
        Parses the response for a given Bedrock LLM response.

        :param response_body: the response body provided from `invoke_model`
        :return: list of str that corresponds to the output in the JSON response
        """
