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

from pytools.api import inheritdoc

from .base import BaseBedrockChat

log = logging.getLogger(__name__)

__all__ = ["TitanBedrockChat"]

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

    def _responses_from_body(self, response_body: dict[str, Any]) -> list[str]:
        """[see superclass]"""
        formatted_responses = []
        for result in response_body["results"]:
            output_text = result["outputText"]
            # Example: stripping unnecessary whitespace and adding a line break
            formatted_text = output_text.strip()
            formatted_text = formatted_text.replace(
                "\n", "\n\n"
            )  # Add extra line breaks for readability

            formatted_responses.append(formatted_text)

        return formatted_responses
