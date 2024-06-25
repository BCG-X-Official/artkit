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
from abc import ABCMeta
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import AbstractSet, Any, Generic, TypeVar, final

from pytools.api import appenddoc, inheritdoc

from ..cache import CacheDB
from ._model import GenAIModel

log = logging.getLogger(__name__)

__all__ = [
    "CachedGenAIModel",
    "GenAIModelAdapter",
]

#
# Type variables
#
# Naming convention used here:
# _ret for covariant type variables used in return positions
# _arg for contravariant type variables used in argument positions

T_GenAIModel_ret = TypeVar("T_GenAIModel_ret", bound=GenAIModel, covariant=True)
T_ModelOutput = TypeVar("T_ModelOutput")


#
# Constants
#


DURATION_MS = 2**-20


#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class GenAIModelAdapter(GenAIModel, Generic[T_GenAIModel_ret], metaclass=ABCMeta):
    """
    A wrapper for a Gen AI system that changes the wrapped LLM system's behavior.

    This is useful for adding functionality to an LLM system without modifying
    its implementation.
    """

    #: The LLM system to wrap.
    model: T_GenAIModel_ret

    def __init__(self, model: T_GenAIModel_ret) -> None:
        """
        :param model: the LLM system to delegate to
        """
        self.model = model

    @property
    @final
    def model_id(self) -> str:
        """[see superclass]"""
        return self.model.model_id

    @final
    def get_model_params(self) -> Mapping[str, Any]:
        """[see superclass]"""
        return self.model.get_model_params()


class CachedGenAIModel(
    GenAIModelAdapter[T_GenAIModel_ret],
    Generic[T_GenAIModel_ret, T_ModelOutput],
    metaclass=ABCMeta,
):
    """
    A wrapper for a Gen AI model that caches responses.

    This is useful for lowering the number of requests to a GenAI model thus reducing
    costs and improving performance, both for testing and for production where
    prompts may be reused, e.g., for generating or augmenting test prompts for
    evaluating a GenAI system.

    The cache is stored in a database.
    """

    #: A mapping of cached system prompts to user prompts to responses.
    cache: CacheDB

    @appenddoc(to=GenAIModelAdapter.__init__)
    def __init__(
        self,
        model: T_GenAIModel_ret,
        *,
        database: Path | str,
    ) -> None:
        """
        :param database: the path to a cache database file, or ``":memory:"`` for an
            in-memory database
        :raises TypeError: if the LLM chat system to be cached is an adapter
        """
        super().__init__(model)

        self.cache = CacheDB(database=database)

    def _get(
        self,
        *,
        prompt: str,
        **model_params: str | int | float | None,
    ) -> list[str] | None:
        """
        Get the cached item for a given prompt.

        :param model_id: the model identifier
        :param prompt: the prompt for which to retrieve the cached responses
        :param model_params: additional model parameters associated with the prompt
        :return: the cached responses, or ``None`` if the prompt is not in the cache
        """
        # First get the value from the cache, to ensure it exists
        return self.cache.get_entry(
            model_id=self.model_id,
            prompt=prompt,
            **{k: _simplify_type(v) for k, v in model_params.items() if v is not None},
        )

    def _put(
        self,
        *,
        prompt: str,
        responses: list[str],
        **model_params: str | int | float | None,
    ) -> list[str]:
        """
        Update the cache with a response or list of responses for a given prompt.

        :param model_id: the identifier of the model that generated the responses
        :param prompt: the prompt for which the responses were generated
        :param responses: the responses to store in the cache
        :param model_params: additional model parameters associated with the prompt
        """
        self.cache.add_entry(
            model_id=self.model_id,
            prompt=prompt,
            responses=responses,
            **{k: _simplify_type(v) for k, v in model_params.items() if v is not None},
        )
        return responses

    def clear_cache(
        self,
        *,
        created_before: datetime | None = None,
        accessed_before: datetime | None = None,
        created_after: datetime | None = None,
        accessed_after: datetime | None = None,
    ) -> None:
        """
        Evict cached responses before or after a certain time threshold.

        :param created_before: if specified, only evict cached responses created before
            this time
        :param accessed_before: if specified, only evict cached responses last accessed
            before this time
        :param created_after: if specified, only evict cached responses created after
            this time
        :param accessed_after: if specified, only evict cached responses last accessed
            after this time
        """
        self.cache.clear(
            model_id=self.model_id,
            created_before=created_before,
            accessed_before=accessed_before,
            created_after=created_after,
            accessed_after=accessed_after,
        )


#
# Auxiliary functions
#


def _simplify_type(value: Any) -> str | int | float | bool:
    """
    Ensure that the given value is a string, integer, float, or boolean.

    If the value is not one of these types, it is converted to a string.

    :param value: the value to check and potentially convert
    :return: the value, potentially converted to a string
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, AbstractSet):
        return str(set(map(_simplify_type, value)))
    elif isinstance(value, Mapping):
        # Convert the dictionary to a string representation that is independent of the
        # order of the dictionary's keys
        return str(
            dict(
                sorted((_simplify_type(k), _simplify_type(v)) for k, v in value.items())
            )
        )
    elif isinstance(value, list):
        return str(list(map(_simplify_type, value)))
    elif isinstance(value, tuple):
        return str(tuple(map(_simplify_type, value)))
    else:
        return str(value)
