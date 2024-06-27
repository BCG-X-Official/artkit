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
Implementation of llm module.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pytools.api import MissingClassMeta, inheritdoc

from ..base import ChatModelConnector
from ..history import ChatHistory

log = logging.getLogger(__name__)

try:
    import boto3
    import botocore
except ImportError:

    class boto3(metaclass=MissingClassMeta, module="boto3"):
        """Placeholder class for missing ``boto3`` class."""

    class botocore(metaclass=MissingClassMeta, module="botocore"):
        """Placeholder class for missing ``botocore`` class."""


__all__ = ["BedrockChat"]

#
# Type variables
#

T_BedrockChat = TypeVar("T_BedrockChat", bound="BedrockChat")

#
# Class declarations
#
log = logging.getLogger(__name__)


@inheritdoc(match="""[see superclass]""")
class BedrockChat(ChatModelConnector[boto3.client]):
    @classmethod
    def get_default_api_key_env(cls) -> str:
        return ""

    def _make_client(self) -> boto3.client:
        return boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

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
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        **model_params: Any,
    ):
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
        self.region = region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        try:
            response = await self.get_client().invoke_model(
                body=message,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as error:
            raise Exception("Boto Core error") from error
        response_body = json.loads(response["body"].read())
        return response_body.get("completion")
