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
Abstract base class to represent connecting to a custom diffusion endpoint.
"""
from abc import ABCMeta, abstractmethod
from contextlib import AsyncExitStack
from typing import Any

from aiohttp import ClientResponse, ClientResponseError, ClientSession

from pytools.api import appenddoc, inheritdoc, subsdoc

from ....util import Image
from ...util import RateLimitException
from ..base import DiffusionModelConnector


@inheritdoc(match="""[see superclass]""")
class CustomDiffusionEndpointConnector(
    DiffusionModelConnector[None], metaclass=ABCMeta
):
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
        url: str | None = None,
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
        self.url = url

    @abstractmethod
    def format_message(self, message: str) -> str:
        """
        This method is responsible for formatting the input to the LLM diffusion system.
        """

    @abstractmethod
    def format_headers(self) -> dict[str, Any]:
        """
        This method is responsible for formatting the headers to the request.
        """

    @abstractmethod
    def format_response(self, response: ClientResponse) -> list[Image]:
        """
        This method is responsible for formatting the response to list of Images.
        """

    async def text_to_image(
        self, text: str, **model_params: dict[str, Any]
    ) -> list[Image]:
        async with AsyncExitStack():
            async with ClientSession(headers=self.format_headers()) as aio_session:
                if self.url is None:
                    raise ValueError("URL must be provided")
                response = await aio_session.post(
                    url=self.url, data=self.format_message(message=text)
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
        return self.format_response(response=response)
