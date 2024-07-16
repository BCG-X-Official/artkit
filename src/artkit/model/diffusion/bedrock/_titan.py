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

import base64
import logging
from typing import Any, TypeVar

from pytools.api import inheritdoc

from ....util import Image
from .base import BaseBedrockDiffusionModel

log = logging.getLogger(__name__)

__all__ = ["TitanBedrockDiffusionModel"]

#
# Type variables
#

T_TitanBedrockDiffusionModel = TypeVar(
    "T_TitanBedrockDiffusionModel", bound="TitanBedrockDiffusionModel"
)

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class TitanBedrockDiffusionModel(BaseBedrockDiffusionModel):
    """
    Implementation of a BaseBedrockDiffusionModel with Titan.
    """

    async def text_to_image(
        self, text: str, **model_params: dict[str, Any]
    ) -> list[Image]:
        """[see superclass]"""
        response_body = await self.get_response_from_model(text)
        images = []
        for img_data in response_body.get("images", []):
            base64_bytes = img_data.encode("ascii")
            img_bytes = base64.b64decode(base64_bytes)
            image = Image(data=img_bytes)
            images.append(image)
        return images
