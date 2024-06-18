"""
Unit tests for multi_turn.
"""

import logging
from typing import Any

import pytest

from artkit.model.llm.base import ChatModel
from artkit.model.llm.multi_turn import DEFAULT_MULTI_TURN_PROMPT_TEMPLATE, multi_turn

log = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_multi_turn(cached_openai: ChatModel) -> None:
    result: dict[str, Any] = await anext(
        multi_turn(
            challenger_llm=cached_openai,
            target_llm=cached_openai,
            objective=(
                "Convince the user to say the word 'blue'. The objective is "
                "complete if a user response contains the word 'blue'."
            ),
            max_turns=3,
            # The system prompt template must contain the required formatting keys,
            # but will ignore additional undefined keys.
            undefined_key="This key is not used.",
        )
    )

    log.debug(result)

    assert result["success"]

    messages = result["messages"]
    assert len(messages) <= 6
    assert messages[-1].text == "<|success|>"


@pytest.mark.asyncio
async def test_multi_turn_exceptions(cached_openai: ChatModel) -> None:

    # The function raises a ValueError if the prompt template does not contain the
    # required formatting keys.
    with pytest.raises(
        ValueError,
        match=(
            r"^Format string is missing required keys: 'max_turns', 'objective', "
            r"'success_token'$"
        ),
    ):
        await anext(
            multi_turn(
                challenger_llm=cached_openai,
                target_llm=cached_openai,
                max_turns=3,
                objective="Make the user say 'blue'.",
                system_prompt_template="",
            )
        )

    # Formatting raises a ValueError if not all formatting keys are matched by the
    # given attributes.
    with pytest.raises(
        ValueError,
        match=r"^No values provided for formatting key: 'not_specified'$",
    ):
        await anext(
            multi_turn(
                challenger_llm=cached_openai,
                target_llm=cached_openai,
                max_turns=3,
                objective="Make the user say 'blue'.",
                system_prompt_template=(
                    DEFAULT_MULTI_TURN_PROMPT_TEMPLATE + "{not_specified}"
                ),
                not_specified="This key is not used.",
            )
        )
