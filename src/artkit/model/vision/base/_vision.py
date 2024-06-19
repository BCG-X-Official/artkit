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
Base implementation of vision model
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar

from ....util import Image
from ...base import ConnectorMixin, GenAIModel
from ...util import retry_with_exponential_backoff

__all__ = [
    "VisionModel",
    "VisionModelConnector",
]


#
# Type variables
#
T_Client = TypeVar("T_Client")


#
# Classes
#


class VisionModel(GenAIModel, metaclass=ABCMeta):
    """
    An abstract vision model.

    Vision models are used to generate text descriptions of images.
    """

    @abstractmethod
    async def image_to_text(
        self,
        image: Image,
        prompt: str | None = None,
        **model_params: Any,
    ) -> list[str]:
        """
        Generates a text response from an image and prompt pair

        :param image: an Image object to pass to the model
        :param prompt: the text prompt to pass to the model
        :param model_params: additional optional parameters for image-to-text generation
        :return: model text response for the input image and prompt pair
        """


class VisionModelConnector(
    VisionModel, ConnectorMixin[T_Client], Generic[T_Client], metaclass=ABCMeta
):
    """
    A vision model that connects to a client.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Apply the retry strategy to the image_to_text method
        cls.image_to_text = (  # type: ignore[method-assign]
            retry_with_exponential_backoff(cls.image_to_text)  # type: ignore
        )
