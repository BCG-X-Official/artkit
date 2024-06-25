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
Implementation of llm module.
"""

from __future__ import annotations

import logging
import os
from abc import ABCMeta, abstractmethod
from collections.abc import Coroutine, Mapping, MutableMapping
from typing import Any, Generic, TypeVar
from weakref import WeakValueDictionary

from pytools.api import inheritdoc
from pytools.asyncio import arun
from pytools.expression import (
    Expression,
    HasExpressionRepr,
    expression_from_init_params,
)

log = logging.getLogger(__name__)


__all__ = [
    "GenAIModel",
    "ConnectorMixin",
]

#
# Type variables
#

T_Client = TypeVar("T_Client")


#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class GenAIModel(HasExpressionRepr, metaclass=ABCMeta):
    """
    A Gen AI model or system.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        The ID of the model to use.
        """

    @abstractmethod
    def get_model_params(self) -> Mapping[str, Any]:
        """
        Get the parameters of the model as a mapping.

        This includes all parameters that influence the model's behavior, but not
        parameters that determine the model itself or are are specific to the client
        such as the model ID or the API key.

        :return: the model parameters
        """

    def to_expression(self) -> Expression:
        """[see superclass]"""

        return expression_from_init_params(self, ignore_null=True)


@inheritdoc(match="""[see superclass]""")
class ConnectorMixin(GenAIModel, Generic[T_Client], metaclass=ABCMeta):
    """
    Mixin class for a Gen AI model that connects to the model's client.

    Connects with a Gen AI model client to interact with the model. The client is
    shared across all instances of the same model ID and API key.

    Connectors can define any number of model parameters that are passed along with
    every request to the model. Default values for these parameters can be set in the
    constructor. Note that parameters with ``None`` values are considered unset and
    will be ignored.
    """

    #: The ID of the model to use.
    _model_id: str

    #: The environment variable that holds the API key.
    api_key_env: str

    #: The initial delay in seconds between client requests.
    initial_delay: float

    #: The base for the exponential backoff.
    exponential_base: float

    #: Whether to add jitter to the delay.
    jitter: bool

    #: The maximum number of retries for client requests.
    max_retries: int

    #: Additional model parameters, passed with every request.
    model_params: dict[str, Any]

    #: Mapping of model ids and API key environment variables to client instances;
    #: used to share clients across instances. Keys are weakly referenced to
    #: allow garbage collection.
    #:
    #: This is a class variable, so it is shared across all instances of the class.
    _clients: MutableMapping[tuple[str, str], _ClientWrapper[T_Client]]

    #: The client used by this instance
    _client: _ClientWrapper[T_Client] | None = None

    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str | None = None,
        initial_delay: float | None = None,
        exponential_base: float | None = None,
        jitter: bool | None = None,
        max_retries: int | None = None,
        **model_params: Any,
    ) -> None:
        """
        :param model_id: the ID of the model to use
        :param api_key_env: the environment variable that holds the API key; if not
          specified, use the default API key environment variable for the model as
          returned by :meth:`get_default_api_key_env`
        :param initial_delay: the initial delay in seconds between client requests
          (defaults to ``1.0``)
        :param exponential_base: the base for the exponential backoff
          (defaults to ``2.0``)
        :param jitter: whether to add jitter to the delay
          (defaults to ``True``)
        :param max_retries: the maximum number of retries for client requests
          (defaults to ``5``)
        :param model_params: additional model parameters, passed with every request;
          parameters with ``None`` values are considered unset and will be ignored
        """
        self._model_id = model_id
        self.api_key_env = (
            self.get_default_api_key_env() if api_key_env is None else api_key_env
        )
        self.initial_delay = initial_delay if initial_delay is not None else 1.0
        self.exponential_base = (
            exponential_base if exponential_base is not None else 2.0
        )
        self.jitter = jitter if jitter is not None else True
        self.max_retries = max_retries if max_retries is not None else 5
        self.model_params = {
            k: v
            for k, v in model_params.items()
            # Filter out unset parameters
            if v is not None
        }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Set up a cache for shared client instances.
        """
        super().__init_subclass__(**kwargs)
        cls._clients = WeakValueDictionary()

    @classmethod
    @abstractmethod
    def get_default_api_key_env(cls) -> str:
        """
        Get the default name of the environment variable that holds the API key.

        :return: the default name of the api key environment variable
        """

    @property
    def model_id(self) -> str:
        """[see superclass]"""
        return self._model_id

    def get_client(self) -> T_Client:
        """
        Get a shared client instance for this connector; return a cached instance if
        for this model's API key if available.

        :return: the shared client instance
        """
        if self._client is None:
            # This instance does not have a client yet; try to get a shared client
            # instance for the API key
            client_key = self.model_id, self.api_key_env
            client = self._clients.get(client_key)
            if client is None:
                # No shared client instance exists yet; create one
                client = self._clients[client_key] = _ClientWrapper(self._make_client())
            # Cache the client instance for this instance
            self._client = client
            return client.client
        else:
            return self._client.client

    def get_api_key(self) -> str:
        """
        Get the API key from the environment variable specified by :attr:`api_key_env`.

        :return: the API key
        :raises ValueError: if the environment variable is not set
        """
        try:
            return os.environ[self.api_key_env]
        except KeyError as e:
            raise ValueError(
                f"The environment variable {self.api_key_env} for the API key of model "
                f"{self.model_id!r} is not set. Please set the environment variable to "
                f"your API key, or revise arg api_key_env to the correct environment "
                f"variable name."
            ) from e

    def get_model_params(self) -> Mapping[str, Any]:
        """[see superclass]"""
        return self.model_params

    @abstractmethod
    def _make_client(self) -> T_Client:
        """
        Create a client instance using the :attr:`api_key_env` attribute of this object.

        :return: the remote client instance
        """

    def to_expression(self) -> Expression:
        """[see superclass]"""

        return expression_from_init_params(
            self, ignore_null=True, kwargs="model_params"
        )


class _ClientWrapper(Generic[T_Client]):
    """
    A wrapper for a client that rate-limits requests, and takes care of releasing the
    clients' resources when they are no longer needed.

    Common API clients (OpenAI, Anthropic, Groq) have their own teardown methods that
    get invoked by a ``__del__`` method in the client class, but they do not handle
    async event loops properly. This wrapper class ensures that the client's resources
    are released correctly when the wrapper is deleted, even if the client's close
    method is asynchronous.
    """

    def __init__(self, client: T_Client) -> None:
        """
        :param client: the client instance to wrap
        """
        self.client = client

    #: The client instance to wrap.
    client: T_Client

    def __del__(self) -> None:
        """
        Release the client's resources when the wrapper is deleted.
        """
        if hasattr(self.client, "close"):
            close_awaitable = self.client.close()
            if isinstance(close_awaitable, Coroutine):
                # The close method is a coroutine; run it in an event loop
                arun(close_awaitable)
