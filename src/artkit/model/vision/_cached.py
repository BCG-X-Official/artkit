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
Implementation of CachedVisionModel.
"""

import hashlib
from typing import Any, Generic, TypeVar

from pytools.api import inheritdoc

from ...util import Image
from ..base import CachedGenAIModel
from .base import VisionModel

__all__ = [
    "CachedVisionModel",
]


#
# Type variables
#
# Naming convention used here:
# _ret for covariant type variables used in return positions
# _arg for contravariant type variables used in argument positions

T_VisionModel_ret = TypeVar("T_VisionModel_ret", bound=VisionModel, covariant=True)


#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class CachedVisionModel(
    CachedGenAIModel[T_VisionModel_ret, list[str]],
    VisionModel,
    Generic[T_VisionModel_ret],
):
    """
    A wrapper for a vision model that caches responses.
    """

    @staticmethod
    def _make_cache_prompt(prompt: str | None, image: Image) -> str:
        """
        Convert the image and prompt into a key for the cache

        :param prompt: the prompt to use to generate the cache key
        :param image: an image Object used to use to generate the cache key
        :return: the cache key as a tuple of strings representing the prompt and image
        """
        # Hash the b64 string representation to decrease memory usage
        hasher = hashlib.sha256()
        hasher.update(image.data)

        # Return a unique key for each prompt and image
        return repr((prompt, hasher.hexdigest()))

    async def image_to_text(
        self,
        image: Image,
        prompt: str | None = None,
        **model_params: Any,
    ) -> list[str]:
        """[see superclass]"""

        # Include model params in cache key
        model_params_merged = {**self.get_model_params(), **model_params}

        # Make cache prompt consisting of image and prompt
        cache_prompt = self._make_cache_prompt(prompt=prompt, image=image)
        response: list[str] | None = self._get(
            prompt=cache_prompt, **model_params_merged
        )

        if response is None:  # pragma: no cover
            return self._put(
                prompt=cache_prompt,
                responses=await self.model.image_to_text(
                    image=image, prompt=prompt, **model_params
                ),
                **model_params_merged,
            )
        else:
            return response
