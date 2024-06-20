import logging
import time
from io import StringIO

from pytest import LogCaptureFixture, fixture

import artkit.api as ak

log = logging.getLogger(__name__)

LOGS_TRUNCATED_MSG = "---- Logs truncated ----"


def test_log_throttling_exceeding_max_messages(
    log_throttling_handler: ak.LogThrottlingHandler,
    logging_test_msg: str,
    log_stream: StringIO,
) -> None:
    # Number of times we emit the log record
    n = log_throttling_handler.max_messages * 2
    for _ in range(n):
        log.warning(logging_test_msg)

    # Get the content from the IO stream
    log_stream_content = log_stream.getvalue().splitlines()

    # Check that the log stream contains the correct number of messages
    assert len(log_stream_content) == (log_throttling_handler.max_messages + 1)

    # Check that the log stream contains the correct messages
    assert log_stream_content == (
        [logging_test_msg for _ in range(log_throttling_handler.max_messages)]
        + [LOGS_TRUNCATED_MSG]
    )


def test_log_throttling_less_than_max_messages(
    log_throttling_handler: ak.LogThrottlingHandler,
    logging_test_msg: str,
    caplog: LogCaptureFixture,
) -> None:
    # Number of times we emit the log record
    n = int(log_throttling_handler.max_messages / 2)
    for _ in range(n):
        log.warning(logging_test_msg)
    same_messages = [
        record for record in caplog.records if record.msg == logging_test_msg
    ]
    assert len(same_messages) == n


def test_restart_logging_after_interval(
    log_throttling_handler: ak.LogThrottlingHandler,
    logging_test_msg: str,
    log_stream: StringIO,
) -> None:
    # Number of times we emit the log record
    n = log_throttling_handler.max_messages * 2
    for _ in range(n):
        log.warning(logging_test_msg)
    log_stream_content = log_stream.getvalue().splitlines()
    assert len(log_stream_content) == (log_throttling_handler.max_messages + 1)

    # Sleep for the interval
    time.sleep(log_throttling_handler.interval + 0.1)
    log.warning(logging_test_msg)
    # Get the content from the IO stream
    log_stream_content = log_stream.getvalue().splitlines()

    # Assert that the last message was sent normally again
    assert log_stream_content == (
        [logging_test_msg for _ in range(log_throttling_handler.max_messages)]
        + [LOGS_TRUNCATED_MSG]
        + [logging_test_msg]
    )


def test_that_changing_args_do_not_matter(
    log_throttling_handler: ak.LogThrottlingHandler,
    log_stream: StringIO,
) -> None:
    test_msg = "Test message %s"

    # Number of times we emit the log record
    n = log_throttling_handler.max_messages * 2
    for i in range(n):
        log.warning(test_msg, i)

    # Get the content from the IO stream
    log_stream_content = log_stream.getvalue().splitlines()

    # Check that the log stream contains the correct number of messages
    assert len(log_stream_content) == (log_throttling_handler.max_messages + 1)

    # Check that the log stream contains the correct messages
    assert log_stream_content == (
        [test_msg % i for i in range(log_throttling_handler.max_messages)]
        + [LOGS_TRUNCATED_MSG]
    )


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################


@fixture
def log_stream() -> StringIO:
    return StringIO()


@fixture
def log_throttling_handler(log_stream: StringIO) -> ak.LogThrottlingHandler:
    _throttling_handler = ak.LogThrottlingHandler(
        # We need to keep track of our own IO stream
        # since pytest capsys fixture struggles with logging
        # calls https://github.com/pytest-dev/pytest/issues/5997
        handler=logging.StreamHandler(log_stream),
        interval=0.5,
        max_messages=5,
    )
    log.addHandler(_throttling_handler)
    return _throttling_handler


@fixture
def logging_test_msg() -> str:
    return "This is a test log record for the log throttling handler."
