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
import os
from abc import ABCMeta
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from artkit.model.llm.history._history import ChatHistory
from pytools.api import MissingClassMeta, appenddoc, inheritdoc, subsdoc

from ...util import RateLimitException
from ..base import ChatModelConnector

logger = logging.getLogger(__name__)

__all__ = ["VertexAIChat"]

try:
    # vertexai is installed as part of pip installing google-cloud-aiplatform.
    # We recommend this instead of installing vertexai individually because the
    # standalone vertexai package is not available in conda and our repo utilizes
    # conda builds/tests

    import vertexai  # type: ignore
    from google.api_core.exceptions import TooManyRequests
    from vertexai.generative_models import (  # type: ignore
        Content,
        GenerativeModel,
        Part,
    )

except ImportError:  # pragma: no cover

    class vertexai(metaclass=MissingClassMeta, module="vertexai"):  # type: ignore
        """Placeholder class for missing ``vertexai`` module."""

    class TooManyRequests(metaclass=MissingClassMeta, module="google.api_core.exceptions"):  # type: ignore
        """Placeholder class for missing ``TooManyRequests`` class."""

    class Content(metaclass=MissingClassMeta, module="vertexai.generative_models"):  # type: ignore
        """Placeholder class for missing ``Content`` class."""

    class GenerativeModel(metaclass=MissingClassMeta, module="vertexai.generative_models"):  # type: ignore
        """Placeholder class for missing ``GenerativeModel`` class."""

    class Part(metaclass=MissingClassMeta, module="vertexai.generative_models"):  # type: ignore
        """Placeholder class for missing ``Part`` class."""


#
# Type variables
#

T_VertexAIChat = TypeVar("T_VertexAIChat", bound="VertexAIChat")

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class VertexAIChat(ChatModelConnector[GenerativeModel], metaclass=ABCMeta):
    """
    Base class for Vertex AI LLMs.
    """

    region: str | None
    gcp_project_id_env: str
    #: If ``False``, disable all accessible safety categories; otherwise, enable them.
    safety: bool

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return ""

    def get_gcp_project_id_env(self) -> str:
        """
        Get the GCP project ID from the environment variable specified by
        :attr:`gcp_project_id_env`.

        :return: the GCP project ID
        :raises EnvironmentError: if the environment variable is not set
        """
        if self.gcp_project_id_env is None:
            raise ValueError("gcp_project_id_env must not be None")

        try:
            return os.environ[self.gcp_project_id_env]
        except KeyError as e:
            raise OSError(
                f"The environment variable {self.gcp_project_id_env} for the GCP project ID "
                f"of model {self.model_id!r} is not set. Please set the environment variable to "
                f"your GCP project ID, or revise the arg gcp_project_id_env to the correct "
                f"environment variable name."
            ) from e

    def _make_client(self) -> GenerativeModel:  # pragma: no cover
        """
        This method handles the authentication and connection to the Vertex AI. It
        assumes you have followed the instructions in
        sphinx/source/user_guide/introduction_to_artkit/connecting_to_genai_models.ipynb
        to setup Application Default Credentials(ADC) to access GCP.
        """
        vertexai.init(project=self.get_gcp_project_id_env(), location=self.region)
        return GenerativeModel(self.model_id, system_instruction=self.system_prompt)

    def get_model_params(self) -> Mapping[str, Any]:
        """[see superclass]"""
        return dict(**super().get_model_params(), safety=self.safety)

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
        gcp_project_id_env: str | None = None,
        safety: bool = True,
        **model_params: Any,
    ) -> None:
        """
        :param region: The specific GCP region to connect to.
        :param gcp_project_id_env: The GCP project ID.
        :param safety: Safety settings for the model.
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
        self.region = region if region else "us-east1"
        self.gcp_project_id_env = (
            "GCP_PROJECT_ID" if gcp_project_id_env is None else gcp_project_id_env
        )
        self.safety = safety

    def _get_safety_settings(self) -> dict[str, str] | None:
        """
        Get safety settings for Gemini.

        :return: a list of dicts describing safety settings by harm category
        """

        if self.safety:
            return None
        else:
            return {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }

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
        """
        Send a message to the Vertex AI model and get a response.

        :param message: The input message to send to the model.
        :param history: The conversation history.
        :param model_params: Additional parameters to pass to the model.
        :return: The model's response.
        """
        async with AsyncExitStack():
            formatted_messages = self._messages_to_vertexai_format(
                message, history=history
            )

            # _make_client creates a new instance of the client connection,
            # while get_client uses a cached instance of the client connection.
            # We use _make_client for Vertex and Gemini because we can only
            # initialize the system prompt when the client is created.

            client = self._make_client()
            try:
                response = await client.generate_content_async(
                    contents=formatted_messages,
                    safety_settings=self._get_safety_settings(),
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
