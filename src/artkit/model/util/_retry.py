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

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar

from ...util import LogThrottlingHandler
from ..base import ConnectorMixin
from ._exception import RateLimitException

__all__ = [
    "retry_with_exponential_backoff",
]


log = logging.getLogger(__name__)
# 5-second interval, max 5 messages
_throttling_handler = LogThrottlingHandler(
    handler=logging.StreamHandler(), interval=5, max_messages=5
)
_throttling_handler.setLevel(logging.WARNING)
log.addHandler(_throttling_handler)

#
# Type variables
#
# Naming convention used here:
# _ret for covariant type variables used in return positions
# _arg for contravariant type variables used in argument positions
#

T_ret = TypeVar("T_ret", covariant=True)
T_Connector = TypeVar("T_Connector", bound=ConnectorMixin[Any])
P = ParamSpec("P")


def retry_with_exponential_backoff(
    func: Callable[Concatenate[T_Connector, P], Coroutine[Any, Any, T_ret]]
) -> Callable[Concatenate[T_Connector, P], Coroutine[Any, Any, T_ret]]:
    """
    Decorator that adds retrying with exponential backoff to a connector method.

    :param func: the function to decorate
    :return: the decorated function
    """

    @wraps(func)
    async def _wrapper(
        self: T_Connector, /, *args: P.args, **kwargs: P.kwargs
    ) -> T_ret:
        """
        Wrapper function for retrying a function with exponential backoff.

        :param self: the connector instance the method is called on
        :param args: the positional arguments passed to the function
        :param kwargs: the keyword arguments passed to the function

        :return: the result of the function
        """
        # Initialize variables
        num_retries = 0
        delay = self.initial_delay

        # Loop until a successful response or max_retries is hit
        # or an uncaught exception is raised
        last_exception = None
        max_retries = self.max_retries
        exponential_base = self.exponential_base
        jitter = self.jitter

        for num_retries in range(1, max_retries + 1):
            try:
                return await func(self, *args, **kwargs)

            # Retry on rate limit errors
            except RateLimitException as e:
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Log retry
                log.warning(
                    "Rate limit exceeded: Retrying after model I/O error "
                    "(%s/%s). Wait for %.2f seconds.",
                    num_retries,
                    max_retries,
                    delay,
                )

                # Sleep for the delay
                await asyncio.sleep(delay)
                last_exception = e

        raise RateLimitException(
            "Rate limit error after max retries."
        ) from last_exception

    if func.__code__ is _wrapper.__code__:  # type: ignore[attr-defined]
        # If the function is already wrapped with our decorator, return it as is.
        # This can happen if a connector's __init_subclass__ method is called
        # multiple times along the MRO.
        return func
    else:
        return _wrapper
