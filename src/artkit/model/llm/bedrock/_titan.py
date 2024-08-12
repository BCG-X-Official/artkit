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
Implementation of Titan Bedrock module.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pytools.api import MissingClassMeta, inheritdoc

from ..history import AssistantMessage, ChatHistory, UserMessage
from .base import BaseBedrockChat

log = logging.getLogger(__name__)

__all__ = ["TitanBedrockChat"]

try:
    from botocore.awsrequest import AWSRequest
except ImportError:

    class AWSRequest(metaclass=MissingClassMeta, module="AWSRequest"):  # type: ignore
        """Placeholder class for missing ``AWSRequest`` class."""


try:
    from httpx import Response

except ImportError:

    class Response(metaclass=MissingClassMeta, module="httpx"):  # type: ignore
        """Placeholder class for missing ``Response`` class."""


#
# Constants
#
TITAN_USER_ROLE = "User: "
TITAN_BOT_ROLE = "Bot: "
TITAN_EOT_TOKEN = "\n"
TITAN_ROLE_MAPPING = {
    UserMessage.ROLE: TITAN_USER_ROLE,
    AssistantMessage.ROLE: TITAN_BOT_ROLE,
}

#
# Type variables
#

T_TitanBedrockChat = TypeVar("T_TitanBedrockChat", bound="TitanBedrockChat")

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class TitanBedrockChat(BaseBedrockChat):
    """
    Implementation of a BaseBedrockChat with Titan LLM as the Foundational Model.
    """

    def build_request_arguments(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> dict[str, Any]:
        """[see superclass]"""
        if self.system_prompt is not None:
            raise ValueError(
                "Titan does not support system prompts. Please remove the system_prompt parameter."
            )
        text = ""
        if history is not None:
            for historic_message in history.messages:
                role = historic_message.role
                message_txt = historic_message.text
                text += role + message_txt + TITAN_EOT_TOKEN

        text += TITAN_USER_ROLE + message + TITAN_EOT_TOKEN
        request = AWSRequest(
            method="POST",
            url=self.endpoint,
            data=message,
            headers={"content-type": "application/json"},
        )
        self.auth.add_auth(request)
        prepped_request = request.prepare()
        headers = dict(prepped_request.headers.items())
        method = "POST"
        return dict(
            method=method,
            url=prepped_request.url,
            headers=headers,
            data=prepped_request.body,
        )

    def parse_httpx_response(self, response: Response) -> list[str]:
        """[see superclass]"""
        response_body = response.json()
        formatted_responses = []
        for result in response_body["results"]:
            output_text = result["outputText"]
            # Example: stripping unnecessary whitespace and adding a line break
            formatted_text = output_text.strip()
            formatted_text = formatted_text.replace(
                "\n", "\n\n"
            )  # Add extra line breaks for readability

            formatted_responses.append(formatted_text)

        return list(formatted_responses)
