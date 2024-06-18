"""
Implementation of OpenAIDiffusion
"""

import base64
from typing import Any

from pytools.api import inheritdoc

from ....util import Image
from ...util import RateLimitException
from ..base import DiffusionModelConnector

try:
    from openai import AsyncOpenAI, RateLimitError
except ImportError:
    from pytools.api import MissingClassMeta

    class AsyncOpenAI(metaclass=MissingClassMeta, module="openai"):  # type: ignore
        """Placeholder class for missing ``AsyncOpenAI`` class."""

    class RateLimitError(metaclass=MissingClassMeta, module="openai"):  # type: ignore
        """Placeholder class for missing ``RateLimitError`` class."""


__all__ = [
    "OpenAIDiffusion",
]


#
# Classes
#


@inheritdoc(match=r"[see superclass]")
class OpenAIDiffusion(DiffusionModelConnector[AsyncOpenAI]):
    """
    An asynchronous OpenAI diffusion model.

    Note that supported parameters vary between models, e.g., the minimum image
    size for DALL-E-3 is 1024x1024.
    """

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return "OPENAI_API_KEY"

    def _make_client(self) -> AsyncOpenAI:  # pragma: no cover
        """[see superclass]"""
        return AsyncOpenAI(api_key=self.get_api_key())

    async def text_to_image(
        self, text: str, **model_params: dict[str, Any]
    ) -> list[Image]:  # pragma: no cover
        """[see superclass]"""
        try:
            response = await self.get_client().images.generate(
                model=self.model_id,
                prompt=text,
                response_format="b64_json",
                **{**self.get_model_params(), **model_params},
            )
        except RateLimitError as e:
            raise RateLimitException from e

        return [
            Image(data=base64.b64decode(d.b64_json))
            for d in response.data
            if d.b64_json
        ]
