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
Implementation of OpenAIVision
"""

from contextlib import AsyncExitStack
from typing import Any

from pytools.api import inheritdoc

from ....util import Image
from ...util import RateLimitException
from ..base import VisionModelConnector

try:
    from openai import AsyncOpenAI, RateLimitError
except ImportError:
    from pytools.api import MissingClassMeta

    class AsyncOpenAI(metaclass=MissingClassMeta, module="openai"):  # type: ignore
        """Placeholder class for missing ``AsyncOpenAI`` class."""

    class RateLimitError(metaclass=MissingClassMeta, module="openai"):  # type: ignore
        """Placeholder class for missing ``RateLimitError`` class."""


__all__ = [
    "OpenAIVision",
]


#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class OpenAIVision(VisionModelConnector[AsyncOpenAI]):
    """
    An asynchronous OpenAI vision model
    """

    #: The default prompt used for image-to-text generation
    DEFAULT_VISION_PROMPT = "What's in this image?"

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return "OPENAI_API_KEY"

    def _make_client(self) -> AsyncOpenAI:  # pragma: no cover
        """[see superclass]"""
        return AsyncOpenAI(api_key=self.get_api_key())

    async def image_to_text(  # pragma: no cover
        self,
        image: Image,
        prompt: str | None = None,
        **model_params: Any,
    ) -> list[str]:
        """[see superclass]"""
        async with AsyncExitStack():
            try:
                response = await self.get_client().chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        self.DEFAULT_VISION_PROMPT
                                        if prompt is None
                                        else prompt
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,"
                                        f"{image.to_b64_string()}",
                                    },
                                },
                            ],
                        }
                    ],
                    **{
                        **self.get_model_params(),
                        # Locally specified model parameters override instance defaults
                        **model_params,
                    },
                )
            except RateLimitError as e:
                raise RateLimitException(
                    "Rate limit exceeded. Please try again later."
                ) from e

        return [
            # Replace None with empty string
            c.message.content or ""
            for c in response.choices
        ]
