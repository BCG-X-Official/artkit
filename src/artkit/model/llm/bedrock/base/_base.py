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
from abc import ABCMeta
from typing import Any, TypeVar

from pytools.api import MissingClassMeta, appenddoc, inheritdoc, subsdoc

from ...base import HTTPXChatConnector

log = logging.getLogger(__name__)

__all__ = ["BaseBedrockChat"]

try:
    from botocore.auth import SigV4Auth
    from botocore.session import Session

except ImportError:

    class SigV4Auth(metaclass=MissingClassMeta, module="SigV4Auth"):  # type: ignore
        """Placeholder class for missing ``SigV4Auth`` class."""

    class Session(metaclass=MissingClassMeta, module="Session"):  # type: ignore
        """Placeholder class for missing ``Session`` class."""


try:
    from httpx import AsyncClient

except ImportError:

    class AsyncClient(metaclass=MissingClassMeta, module="httpx"):  # type: ignore
        """Placeholder class for missing ``AsyncClient`` class."""


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
class BaseBedrockChat(HTTPXChatConnector, metaclass=ABCMeta):
    """
    Base class for Bedrock LLMs.
    """

    region: str | None

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return ""

    @subsdoc(
        # The pattern matches the row defining model_params, and move it to the end
        # of the docstring.
        pattern=r"(:param model_params: .*\n)((:?.|\n)*\S)(\n|\s)*",
        replacement=r"\2\1",
    )
    @appenddoc(to=HTTPXChatConnector.__init__)
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
        httpx_client_kwargs: dict[str, Any] | None = None,
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
            httpx_client_kwargs=httpx_client_kwargs,
            **model_params,
        )

        self.region = region if region else "us-east-1"
        self.session = Session()
        self.credentials = self.session.get_credentials()
        self.auth = SigV4Auth(self.credentials, "bedrock", self.region)
        self.endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}/invoke"
