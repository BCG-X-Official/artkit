"""
Implementation of CachedLLM.
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from copy import copy
from typing import Any, Generic, TypeVar

from pytools.api import inheritdoc

from ...base import GenAIModelAdapter
from ._llm import ChatModel, CompletionModel

log = logging.getLogger(__name__)

__all__ = [
    "ChatModelAdapter",
    "CompletionModelAdapter",
]

#
# Type variables
#
# Naming convention used here:
# _ret for covariant type variables used in return positions
# _arg for contravariant type variables used in argument positions

T_CompletionModel_ret = TypeVar(
    "T_CompletionModel_ret", bound=CompletionModel, covariant=True
)
T_ChatModel_ret = TypeVar("T_ChatModel_ret", bound=ChatModel, covariant=True)
T_ChatModelAdapter = TypeVar("T_ChatModelAdapter", bound="ChatModelAdapter[ChatModel]")


#
# Constants
#


DURATION_MS = 2**-20


#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class CompletionModelAdapter(
    CompletionModel,
    GenAIModelAdapter[T_CompletionModel_ret],
    Generic[T_CompletionModel_ret],
    metaclass=ABCMeta,
):
    """
    A wrapper for an LLM text generator.
    """

    async def get_completion(
        self, *, prompt: str, **model_params: dict[str, Any]
    ) -> str:
        """[see superclass]"""
        return await self.model.get_completion(prompt=prompt, **model_params)


@inheritdoc(match="""[see superclass]""")
class ChatModelAdapter(
    ChatModel,
    GenAIModelAdapter[T_ChatModel_ret],
    Generic[T_ChatModel_ret],
    metaclass=ABCMeta,
):
    """
    A wrapper for an LLM system that changes the wrapped LLM system's behavior.

    This is useful for adding functionality to an LLM system without modifying
    its implementation.
    """

    @property
    def system_prompt(self) -> str | None:
        """[see superclass]"""
        return self.model.system_prompt

    def with_system_prompt(
        self: T_ChatModelAdapter, system_prompt: str
    ) -> T_ChatModelAdapter:
        """[see superclass]"""
        llm_copy: T_ChatModelAdapter = copy(self)
        llm_copy.model = self.model.with_system_prompt(system_prompt=system_prompt)
        return llm_copy


#
# Auxiliary functions
#
