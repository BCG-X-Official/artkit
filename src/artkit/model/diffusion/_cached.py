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
Implementation of the ``CachedDiffusionModel`` class.
"""

from typing import Any, Generic, TypeVar

from pytools.api import inheritdoc

from ...util import Image
from ..base import CachedGenAIModel
from .base import DiffusionModel

__all__ = [
    "CachedDiffusionModel",
]

#
# Type variables
#
# Naming convention used here:
# _ret for covariant type variables used in return positions
# _arg for contravariant type variables used in argument positions


T_DiffusionModel_ret = TypeVar(
    "T_DiffusionModel_ret", bound=DiffusionModel, covariant=True
)


#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class CachedDiffusionModel(
    CachedGenAIModel[T_DiffusionModel_ret, list[str]],
    DiffusionModel,
    Generic[T_DiffusionModel_ret],
):
    """
    A wrapper around a diffusion model to cache results
    """

    async def text_to_image(
        self, text: str, **model_params: dict[str, Any]
    ) -> list[Image]:
        """[see superclass]"""

        # Include model parameters in cache key
        model_params_merged = {
            "_type": "diffusion",
            **self.get_model_params(),
            **model_params,
        }

        # Try to get images from cache
        response: list[str] | None = self._get(prompt=text, **model_params_merged)
        if response is None:
            # Get Image objects from model and cache b64 string
            images: list[Image] = await self.model.text_to_image(
                text=text, **model_params
            )
            self._put(
                prompt=text,
                responses=[
                    img.data.decode(
                        # decode bytes to a string, without changing the data
                        "latin1"
                    )
                    for img in images
                ],
                **model_params_merged,
            )
            return images
        else:
            # Decode and return the images from the cache
            return [
                Image(
                    # decode a string back to bytes
                    data=d.encode("latin1")
                )
                for d in response
            ]
