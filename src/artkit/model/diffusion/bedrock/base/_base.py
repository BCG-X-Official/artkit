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
Bedrock Diffusion systems.
"""
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from pytools.api import MissingClassMeta, appenddoc, inheritdoc, subsdoc

from .....util import Image
from ....util import RateLimitException
from ...base import DiffusionModelConnector

log = logging.getLogger(__name__)

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


__all__ = ["BaseBedrockDiffusionModel"]

#
# Type variables
#

T_BaseBedrockDiffusionModel = TypeVar(
    "T_BaseBedrockDiffusionModel", bound="BaseBedrockDiffusionModel"
)

#
# Class declarations
#
log = logging.getLogger(__name__)


@inheritdoc(match="""[see superclass]""")
class BaseBedrockDiffusionModel(DiffusionModelConnector[None], metaclass=ABCMeta):
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
        """[see superclass]"""
        return None

    @subsdoc(
        # The pattern matches the row defining model_params, and move it to the end
        # of the docstring.
        pattern=r"(:param model_params: .*\n)((:?.|\n)*\S)(\n|\s)*",
        replacement=r"\2\1",
    )
    @appenddoc(to=DiffusionModelConnector.__init__)
    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str | None = None,
        initial_delay: float | None = None,
        exponential_base: float | None = None,
        jitter: bool | None = None,
        max_retries: int | None = None,
        region: str,
        **model_params: Any,
    ) -> None:
        """
        Initialize the BaseBedrockDiffusionModel.

        :param region: The AWS region.
        :raises CredentialsNotFoundError: if unable to find AWS Credentials.
        """
        super().__init__(
            model_id=model_id,
            api_key_env=api_key_env,
            initial_delay=initial_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            max_retries=max_retries,
            model_params=model_params,
        )
        self.region = region if region else "us-east-1"
        self.session = Session()
        self.credentials = self.session.get_credentials()
        self.auth = SigV4Auth(self.credentials, "bedrock", self.region)
        self.endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}/invoke"

    async def get_response_from_model(self, body: str) -> dict[str, Any]:
        """
        Gets a response from the model

        :param body: The body of the request.
        :return: The response from the model.
        """
        request = AWSRequest(
            method="POST",
            url=self.endpoint,
            data=body,
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

        response_body: dict[str, Any] = await response.json()
        return response_body

    @abstractmethod
    async def text_to_image(
        self, text: str, **model_params: dict[str, Any]
    ) -> list[Image]:
        """
        Convert JSON text payload to images using the diffusion model.

        :param text: The payload for the specific diffusion model.
        :param model_params: Additional model parameters.
        :return: List of Image that corresponds to the output in the model response.
        """
        pass
