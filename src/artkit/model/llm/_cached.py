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
Implementation of CachedLLM.
"""

from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

from pytools.api import inheritdoc

from ..base import CachedGenAIModel
from .base import ChatModel, ChatModelAdapter, CompletionModel, CompletionModelAdapter
from .history import ChatHistory

log = logging.getLogger(__name__)

__all__ = [
    "CachedCompletionModel",
    "CachedChatModel",
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


@inheritdoc(match="""[see superclass]""")
class CachedCompletionModel(
    CachedGenAIModel[T_CompletionModel_ret, list[str]],
    CompletionModelAdapter[T_CompletionModel_ret],
    Generic[T_CompletionModel_ret],
):
    """
    A wrapper for a chat model, caching responses.

    This is useful for testing and in production where repeated requests to the model
    are made, thus reducing costs and improving performance.
    """

    async def get_completion(
        self, *, prompt: str, **model_params: dict[str, Any]
    ) -> str:
        """[see superclass]"""

        model_params_merged = {
            # Add the completion flag to the model params to avoid collisions with
            # chat models
            "_type": "completion",
            **self.get_model_params(),
            **model_params,
        }
        response: list[str] | None = self._get(prompt=prompt, **model_params_merged)

        if response is None:  # pragma: no cover
            return self._put(
                prompt=prompt,
                responses=[
                    await self.model.get_completion(prompt=prompt, **model_params)
                ],
                **model_params_merged,
            )[0]
        else:
            return response[0]


@inheritdoc(match="""[see superclass]""")
class CachedChatModel(
    CachedGenAIModel[T_ChatModel_ret, list[str]],
    ChatModelAdapter[T_ChatModel_ret],
    Generic[T_ChatModel_ret],
):
    """
    A wrapper for a chat model, caching responses.

    This is useful for testing and in production where repeated requests to the model
    are made, thus reducing costs and improving performance.


    Example:

    .. code-block:: python

        import artkit.api as ak

        cached_openai_llm = ak.CachedChatModel(
            model=ak.OpenAIChat(model_id="gpt-3.5-turbo"),
            database="./cache/example_cache_gpt3.db"
        )
    """

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""

        model_params_merged = {**self.get_model_params(), **model_params}
        # Add the chat flag to the model params to avoid collisions
        # with completion models

        # Add the system prompt and chat history to the model params
        # to ensure consistent caching behavior
        if self.system_prompt is not None:
            model_params_merged["_system_prompt"] = self.system_prompt
        if history is not None:
            for i, msg in enumerate(history.get_messages()):
                model_params_merged[f"_history_{i}"] = f"[{msg.role}]\n{msg.text}"

        response: list[str] | None = self._get(prompt=message, **model_params_merged)

        if response is None:  # pragma: no cover
            return self._put(
                prompt=message,
                responses=await self.model.get_response(
                    message, history=history, **model_params
                ),
                **model_params_merged,
            )
        else:
            return response
