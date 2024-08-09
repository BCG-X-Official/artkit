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
VertexAI LLM systems.
"""
from __future__ import annotations

import logging
from abc import ABCMeta
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from artkit.model.llm.history._history import ChatHistory
from pytools.api import MissingClassMeta, appenddoc, inheritdoc

from ....util import RateLimitException
from ...base import ChatModelConnector

logger = logging.getLogger(__name__)

__all__ = ["BaseVertexAIChat"]

try:
    import vertexai
    from google.api_core.exceptions import TooManyRequests
    from vertexai.generative_models import Content, GenerativeModel, Part

except ImportError:

    class GenerativeModelError(metaclass=MissingClassMeta, module="GenerativeModel"):
        """Placeholder class for missing ``GenerativeModel`` class."""


__all__ = ["BaseVertexAIChat"]

#
# Type variables
#

T_BaseVertexAIChat = TypeVar("T_BaseVertexAIChat", bound="BaseVertexAIChat")

#
# Class declarations
#
logger = logging.getLogger(__name__)


@inheritdoc(match="""[see superclass]""")
class BaseVertexAIChat(ChatModelConnector[GenerativeModel], metaclass=ABCMeta):
    """
    Base class for Vertex AI LLMs.
    """

    region: str | None
    gcp_project_id: str | None

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return ""

    def _make_client(self) -> GenerativeModel:  # pragma: no cover
        vertexai.init(project=self.gcp_project_id, location=self.region)
        return GenerativeModel(self.model_id)

    @appenddoc(to=ChatModelConnector.__init__)
    def __init__(
        self,
        *,
        model_id: str,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        system_prompt: str | None = None,
        region: str | None = None,
        gcp_project_id: str,
        **model_params: Any,
    ) -> None:
        """
        :param region: The specific GCP region to connect to.
        """
        super().__init__(
            model_id=model_id,
            api_key_env=None,
            initial_delay=initial_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            max_retries=max_retries,
            system_prompt=system_prompt,
            **model_params,
        )
        self.region = region if region else "us-east1"
        self.gcp_project_id = gcp_project_id
        self.endpoint = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.gcp_project_id}/locations/{self.region}/publishers/google/models/{self.model_id}:generateContent"

    @staticmethod
    def _messages_to_vertexai_format(
        user_message: str, *, history: ChatHistory | None = None
    ) -> list[Content]:
        """
        Get the messages to send to the Vertex AI LLM, based on the given user prompt
        and chat history, and the system prompt for this LLM.

        :param user_message: the user prompt to send to the Vertex AI LLM
        :param history: the chat history to include in the messages (optional)
        :return: the messages object, in the format expected by the Vertex AI API
        """
        messages = []

        if history is not None:
            for message in history:
                messages.append(
                    Content(role="user", parts=[Part.from_text(content=message)])
                )

        messages.append(Content(role="user", parts=[Part.from_text(user_message)]))
        return messages

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """Send a message to the Vertex AI model and get a response."""
        async with AsyncExitStack():
            formatted_messages = self._messages_to_vertexai_format(
                message, history=history
            )
            client = self._make_client()
            try:
                response = await client.generate_content_async(
                    contents=formatted_messages
                )

            except TooManyRequests as e:
                raise RateLimitException(
                    "Rate limit exceeded. Please try again later."
                ) from e

        return [
            part.text
            for candidate in response.candidates
            for part in candidate.content.parts
        ]
