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
Implementation of ``LogThrottlingHandler``.
"""

import logging
import time
from collections import defaultdict

__all__ = [
    "LogThrottlingHandler",
]


class LogThrottlingHandler(logging.Handler):
    """
    A logging handler that limits the number of messages during a given time interval.

    This handler can be used with Python's native :mod:`logging` module to throttle
    the maximum number of messages sent during a given time interval.
    This is useful for preventing log spamming in high-throughput applications.

    To follow the native logging flow
    (https://docs.python.org/3/howto/logging.html#logging-flow),
    functionality is implemented across the :meth:`.filter` and :meth:`.emit` methods.

    Example:

    .. code-block:: python

        import logging
        from artkit.util import LogThrottlingHandler

        console_handler = logging.StreamHandler()
        throttling_handler = LogThrottlingHandler(
            console_handler,
            interval=5,
            max_messages=5
        )
        logger = logging.getLogger()
        logger.addHandler(throttling_handler)
    """

    def __init__(
        self, handler: logging.Handler, interval: float, max_messages: int
    ) -> None:
        """
        :param handler: the handler to wrap
        :param interval: the minimum interval in seconds between log messages
          with the same message
        :param max_messages: the maximum number of messages to log within the interval
        """
        super().__init__()
        self.handler = handler
        self.interval = interval
        self.max_messages = max_messages
        self.log_counts: defaultdict[str, tuple[int, float, list[str]]] = defaultdict(
            lambda: (0, 0, [])
        )

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter a log record based on the throttling settings.

        :param record: the log record to filter
        :return: ``True`` if the max messages are not exceeded
          within the time interval, otherwise ``False``
        """

        # if any other filter was registered and returns False, return False
        if not super().filter(record):
            return False

        count, last_log_time, buffer = self.log_counts[record.msg]

        current_time = time.time()
        if current_time - last_log_time > self.interval:
            count = 0
            last_log_time = current_time
            buffer = []

        self.log_counts[record.msg] = (count, last_log_time, buffer)

        if count > self.max_messages:
            return False
        return True

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record if the max messages are not exceeded within the time interval.

        :param record: the log record to emit
        """
        count, last_log_time, buffer = self.log_counts[record.msg]

        if count < self.max_messages:
            self.handler.emit(record)
            buffer.append(record.msg)
            count += 1
        elif count == self.max_messages:
            ellipsis_record = self._create_ellipsis_record(record)
            self.handler.emit(ellipsis_record)
            count += 1

        self.log_counts[record.msg] = (count, last_log_time, buffer)

    @staticmethod
    def _create_ellipsis_record(record: logging.LogRecord) -> logging.LogRecord:
        """
        Create a log record with a custom message to indicate throttling.

        :param record: the original log record
        :return: a new log record with the custom message message
        """
        return logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg="---- Logs truncated ----",
            args=(),
            exc_info=record.exc_info,
        )
