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
Implementation of the `coerce_json_format` function, which uses an LLM to coerce a
JSON-like string into the specified dictionary format. The function is used to
validate the JSON format of the result and coerce it into the expected format.
"""

import logging
from collections.abc import Callable
from json import JSONDecodeError
from json import loads as json_loads
from typing import Any

from artkit.model.llm.base import ChatModel

log = logging.getLogger(__name__)

__all__ = [
    "parse_json_autofix",
]


#
# Constants
#

# System prompt to fix JSON syntax errors
_JSON_REPAIR_SYNTAX_SYSTEM_PROMPT = """\
You must correct the syntax in a JSON string provided by the user. \
Take your time to ensure the syntax is correct. \
Return only the corrected JSON, with no additional formatting or context."""

# System prompt to fix JSON validation error
_JSON_REPAIR_CONTENT_SYSTEM_PROMPT = """\
You must correct the following ERROR in a JSON string provided by the user. \
Take your time to ensure the resulting JSON corrects the error and has correct syntax. \
Return only the corrected JSON, with no additional formatting or context.

ERROR: {error}"""


#
# Functions
#


async def parse_json_autofix(
    json: str,
    *,
    model: ChatModel,
    validator: Callable[[Any], str | None] | None = None,
) -> Any:
    """
    Convert a string in JSON format to a Python object, using the given chat model to
    fix the JSON format if parsing fails.

    An optional validator function can be provided to validate the resulting object,
    triggering another attempt to fix the response if the object is invalid. The
    function must return a string with an error description if the object is invalid,
    and ``None`` if the object is valid. The error description will be passed to the
    chat model as part of the instructions to fix the JSON format.

    :param json: the JSON string to convert to a dictionary
    :param model: the chat model used to repair the JSON format, if necessary
    :param validator: an optional function to validate the resulting object, returning
        ``True`` if the object is valid, and ``False`` otherwise
    :return: the resulting object
    """

    # Attempt to load the JSON string
    try:
        parsed = json_loads(json)
    except JSONDecodeError:
        # Parsing failed
        log.warning(f"Attempting to fix malformed JSON:\n{json}")
        # Ask the model to fix the JSON syntax; we only use the first response in case
        # there are multiple responses
        json_fixed = (
            await model.with_system_prompt(
                _JSON_REPAIR_SYNTAX_SYSTEM_PROMPT
            ).get_response(message=json)
        )[0]
        try:
            parsed = json_loads(json_fixed)
        except JSONDecodeError as e:
            # Parsing failed again: give up
            raise ValueError(
                f"Failed to fix malformed JSON with error {str(e)!r}:\n{json_fixed}"
            )

        # Parsing succeeded: use the fixed JSON
        json = json_fixed

    # Validate the resulting object
    if validator is None:
        # No validation function was provided
        return parsed

    # We have a validation function: check if the object is valid
    error = validator(parsed)
    if error is None:
        # Validation was successful
        return parsed

    # The object is invalid
    log.warning(
        f"Attempting to fix failed validation with error {error!r} for:\n{json}"
    )
    # Ask the model to fix the JSON content using the error message for guidance
    json_fixed = (
        await model.with_system_prompt(
            _JSON_REPAIR_CONTENT_SYSTEM_PROMPT.format(
                # We insert the error message into the system prompts
                error=error
            )
        ).get_response(message=json)
    )[0]

    try:
        # Attempt to load the fixed JSON
        parsed = json_loads(json_fixed)

    except JSONDecodeError as e:
        # Parsing failed
        log.warning(
            f"Malformed JSON with error {str(e)!r} after attempting to fix "
            f"validation error:\n{json_fixed}"
        )

    else:
        # Re-validate the fixed object
        new_error = validator(parsed)
        if new_error is None:
            # Fixed JSON is valid: return it
            return parsed

        # The fixed JSON is still invalid
        log.warning(
            f"New validation error {new_error!r} after attempting to fix the "
            f"previous error:\n{json_fixed}"
        )

    # Parsing failed: give up
    raise ValueError(f"Failed to fix JSON after validation error {error!r}:\n{json}")
