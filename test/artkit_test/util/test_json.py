"""
Test JSON utility functions.
"""

from __future__ import annotations

import logging

import pytest

from artkit.model.llm.base import ChatModel
from artkit.model.llm.util import parse_json_autofix

log = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_json_autofix(cached_openai: ChatModel) -> None:
    """
    Test JSON auto-fixing.
    """

    # Test valid JSON
    assert await parse_json_autofix(
        '{"a": 1, "b": 2}',
        model=cached_openai,
    ) == dict(a=1, b=2)

    # Test valid JSON with successful validator
    assert await parse_json_autofix(
        '{"a": 1, "b": 2}',
        model=cached_openai,
        validator=lambda _: None,
    ) == dict(a=1, b=2)

    # Test invalid JSON
    assert await parse_json_autofix(
        '{"a": 1, "b": 2',
        model=cached_openai,
    ) == dict(a=1, b=2)

    # Test valid JSON with failing validator
    assert await parse_json_autofix(
        '{"a": 1, "b": 2}',
        model=cached_openai,
        validator=lambda d: None if d["b"] == 3 else "b should be 3",
    ) == dict(a=1, b=3)

    # Test valid JSON with wrong syntax after attempting to fix it
    with pytest.raises(
        ValueError,
        match=(
            r"^Failed to fix malformed JSON with error 'Expecting value: "
            r"line 1 column 1 \(char 0\)':"
        ),
    ):
        await parse_json_autofix(
            "f = lambda x: x + 1",
            model=cached_openai,
        )

    # Test valid JSON with wrong syntax after attempting to fix it
    with pytest.raises(
        ValueError,
        match=(
            r"^Failed to fix JSON after validation error 'replace with a Python lambda "
            r"function that returns 1':\n1$"
        ),
    ):
        await parse_json_autofix(
            "1",
            model=cached_openai,
            validator=lambda _: "replace with a Python lambda function that returns 1",
        )

    # Test validation failure after attempting to fix it
    with pytest.raises(
        ValueError,
        match=(
            r"Failed to fix JSON after validation error \'b should be 3\':\n"
            r'{"a": 1, "b": 2}'
        ),
    ):
        await parse_json_autofix(
            '{"a": 1, "b": 2}',
            model=cached_openai,
            validator=lambda _: "b should be 3",
        )
