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
Implementation of GeminiVision
"""

from collections.abc import Iterator
from contextlib import AsyncExitStack
from typing import Any

from pytools.api import inheritdoc

from ....util import Image
from ...util import RateLimitException
from ..base import VisionModelConnector

try:  # pragma: no covers
    from google.api_core.exceptions import TooManyRequests
    from google.generativeai import GenerationConfig, GenerativeModel, configure
    from google.generativeai.types import AsyncGenerateContentResponse
except ImportError:
    from pytools.api import MissingClassMeta

    class GenerativeModel(  # type: ignore[no-redef]
        metaclass=MissingClassMeta, module="google.generativeai"
    ):
        """Placeholder class for missing ``GenerativeModel`` class."""

    class GenerationConfig(  # type: ignore[no-redef]
        metaclass=MissingClassMeta, module="google.generativeai"
    ):
        """Placeholder class for missing ``GenerationConfig`` class."""

    class AsyncGenerateContentResponse(  # type: ignore[no-redef]
        metaclass=MissingClassMeta, module="google.generativeai.types"
    ):
        """Placeholder class for missing ``AsyncGenerateContentResponse`` class."""

    class configure(  # type: ignore[no-redef]
        # pragma: no cover
        metaclass=MissingClassMeta,
        module="google.generativeai",
    ):
        """Placeholder class for missing ``configure`` function."""


__all__ = [
    "GeminiVision",
]


#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class GeminiVision(VisionModelConnector[GenerativeModel]):
    """
    An asynchronous Gemini vision model
    """

    #: The default prompt used for image-to-text generation
    DEFAULT_VISION_PROMPT = "What's in this image?"

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return "GEMINI_API_KEY"

    def _make_client(self) -> GenerativeModel:  # pragma: no cover
        """[see superclass]"""
        configure(api_key=self.get_api_key())
        return GenerativeModel(self.model_id)

    async def image_to_text(  # pragma: no cover
        self,
        image: Image,
        prompt: str | None = None,
        **model_params: Any,
    ) -> list[str]:
        """[see superclass]"""
        async with AsyncExitStack():
            try:
                config = GenerationConfig(**{**self.model_params, **model_params})

                response = await self.get_client().generate_content_async(
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": "text/plain",
                                        "data": (
                                            self.DEFAULT_VISION_PROMPT
                                            if prompt is None
                                            else prompt
                                        ).encode("utf-8"),
                                    }
                                },
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": image.to_b64_string().encode("utf-8"),
                                    }
                                },
                            ],
                        }
                    ],
                    generation_config=config,
                )
            except TooManyRequests as e:
                raise RateLimitException(
                    "Rate limit exceeded. Please try again later."
                ) from e

        return list(self._text_from_response(response))

    @staticmethod
    def _text_from_response(
        response: AsyncGenerateContentResponse,
    ) -> Iterator[str]:
        """
        Get the text from the given response.

        :param response: the response to process
        :return: the text of the responses
        """
        for candidate in response.candidates:
            yield candidate.content.parts[0].text
