"""
Implementation of ``LLMMock``.
"""

from __future__ import annotations

from typing import Any

from artkit.model.llm.base import ChatModel
from artkit.model.llm.history import ChatHistory

__all__ = [
    "MockChatModel",
]


class MockChatModel(ChatModel):  # pragma: no cover
    """
    A mock LLM system that always returns the same response.
    """

    #: The response that this mock LLM system should always return.
    responses: list[str]

    def __init__(self, responses: list[str]) -> None:
        """
        :param responses: the response that this mock LLM system should always return
        """
        self.responses = responses

    @property
    def model_id(self) -> str:
        """
        Always ``"mock"``.
        """
        return "mock"

    @property
    def system_prompt(self) -> str | None:
        """
        Always ``None``.
        """
        return None

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """
        Return the fixed responses of this mock LLM system.

        :param message: ignored
        :param history: ignored
        :param model_params: ignored
        :return: the fixed responses
        """
        return self.responses

    def with_system_prompt(self, system_prompt: str) -> MockChatModel:
        """
        Return a new mock LLM system with the given system prompt.

        :param system_prompt: ignored
        :return: ``self``, ignoring the system prompt
        """
        return self

    def get_model_params(self) -> dict[str, str]:
        """
        Return an empty dictionary.

        :return: an empty dictionary
        """
        return {}
