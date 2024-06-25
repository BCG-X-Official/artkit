"""
Cache tests.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pytest

from artkit.model.cache import CacheDB

log = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def cache() -> CacheDB:
    """
    Fixture for the CacheDB class.
    """
    return CacheDB(database=":memory:")


def test_cache_db(cache: CacheDB) -> None:
    """
    Test the CacheDB class.
    """

    # Constants: model identifiers
    model_a = "model a"
    model_b = "model b"

    # Constants: prompts
    prompt_a = "prompt a"
    prompt_b = "prompt b"

    # Constants: responses
    response_single = "test_response"
    response_multi = ["test_response 1", "test_response 2"]

    # Constants: model parameters
    params_a: dict[str, str | int | float | bool] = dict(
        a_int=2, a_float=2.0, a_str="2", a_bool=True
    )
    params_b: dict[str, str | int | float | bool] = dict(
        b_int=3, b_float=3.0, b_str="3", b_bool=False
    )
    params_ab = {**params_a, **params_b}

    # Test the cache database
    assert cache.conn is not None

    # Try to get a non-existent model response
    assert cache.get_entry(model_id=model_a, prompt=prompt_a) is None

    # Add a new single response
    cache.add_entry(
        model_id=model_a,
        prompt=prompt_a,
        responses=response_single,
    )

    # Get the single response
    assert cache.get_entry(model_id=model_a, prompt=prompt_a) == [response_single]

    # Fail if we add model parameters that are not associated with the cache entry
    assert cache.get_entry(model_id=model_a, prompt=prompt_a, **params_a) is None

    # Fail if we use a different model identifier
    assert cache.get_entry(model_id=model_b, prompt=prompt_a) is None

    # Add a multi response with model parameters
    cache.add_entry(
        model_id=model_a, prompt=prompt_b, responses=response_multi, **params_a
    )

    # Get the multi response with model parameters
    assert (
        cache.get_entry(model_id=model_a, prompt=prompt_b, **params_a) == response_multi
    )

    # Fail if we use model parameters that are not associated with the cache entry
    assert cache.get_entry(model_id=model_a, prompt=prompt_b, **params_b) is None

    # Fail if we use additional model parameters that are not associated with the cache
    # entry
    assert cache.get_entry(model_id=model_a, prompt=prompt_b, **params_ab) is None

    # Fail if we do not add model parameters that are associated with the cache entry
    assert cache.get_entry(model_id=model_a, prompt=prompt_b) is None


def test_cache_clearing_before(cache: CacheDB) -> None:
    """
    Test clearing the cache before a given datetime.
    """

    cache.clear()

    # Constants: model identifiers
    model_a = "model a"
    model_b = "model b"

    # We create 4 cache entries
    n_entries = 4

    # Constants: prompts
    prompts = [f"prompt {i}" for i in range(n_entries)]

    # Constants: responses
    responses = [[f"response {i}"] for i in range(n_entries)]

    # Constants: model parameters
    params: list[dict[str, str | int | float | bool]] = [
        dict(a_int=i, a_text=f"parameter {i}", a_float=i + 0.5, a_text_shared="shared")
        for i in range(n_entries)
    ]

    # Create a new cache database
    cache.add_entry(
        model_id=model_a, prompt=prompts[0], responses=responses[0], **params[0]
    )
    time.sleep(1)
    timestamp_0 = now()
    time.sleep(1)

    # Create additional cache entries
    for i in range(1, n_entries):
        cache.add_entry(
            model_id=model_a, prompt=prompts[i], responses=responses[i], **params[i]
        )

    # Check that all entries are in the cache
    for i in range(n_entries):
        print(f"Checking entry {i}")
        assert (
            cache.get_entry(
                model_id=model_a,
                prompt=prompts[i],
                **params[i],
            )
            == responses[i]
        )

    # Clear the entry with the oldest creation time
    cache.clear(created_before=timestamp_0)

    # Check that entry 0 is not in the cache, but the others are
    assert cache.get_entry(model_id=model_a, prompt=prompts[0], **params[0]) is None
    for i in range(1, n_entries):
        assert (
            cache.get_entry(model_id=model_a, prompt=prompts[i], **params[i])
            == responses[i]
        )

    time.sleep(1)
    timestamp_1 = now()

    # Access entry 1
    assert (
        cache.get_entry(model_id=model_a, prompt=prompts[1], **params[1])
        == responses[1]
    )

    # Clear all entries accessed before timestamp_1
    cache.clear(accessed_before=timestamp_1, model_id=model_a)

    # Confirm only entry 1 is in the cache
    for i in range(n_entries):
        entry = cache.get_entry(model_id=model_a, prompt=prompts[i], **params[i])
        if i == 1:
            assert entry == responses[i]
        else:
            assert entry is None

    # Clear the cache
    cache.clear(model_id=model_b)

    # Confirm that we still have entry 1 in the cache
    assert (
        cache.get_entry(model_id=model_a, prompt=prompts[1], **params[1])
        == responses[1]
    )
    assert cache.count_entries() == {model_a: 1}

    # Clear the cache agai, but this time for model_a
    cache.clear(model_id=model_a)

    # Check that all entries are cleared
    for i in range(n_entries):
        assert cache.get_entry(model_id=model_a, prompt=prompts[i], **params[i]) is None
    assert cache.count_entries() == {}


def test_cache_clearing_after(cache: CacheDB) -> None:
    """
    Test clearing the cache after a given datetime.
    """

    cache.clear()

    # Constants: model identifiers
    model_a = "model a"
    model_b = "model b"

    # We create 4 cache entries
    n_entries = 4

    # Constants: prompts
    prompts = [f"prompt {i}" for i in range(n_entries)]

    # Constants: responses
    responses = [[f"response {i}"] for i in range(n_entries)]

    # Constants: model parameters
    params: list[dict[str, str | int | float | bool]] = [
        dict(a_int=i, a_text=f"parameter {i}", a_float=i + 0.5, a_text_shared="shared")
        for i in range(n_entries)
    ]

    # Create a new cache database
    for i in range(0, n_entries - 1):
        cache.add_entry(
            model_id=model_a, prompt=prompts[i], responses=responses[i], **params[i]
        )

    time.sleep(1)
    timestamp_0 = now()
    time.sleep(1)
    # Create last cache entry
    cache.add_entry(
        model_id=model_a, prompt=prompts[-1], responses=responses[-1], **params[-1]
    )

    # Check that all entries are in the cache
    for i in range(n_entries):
        print(f"Checking entry {i}")
        assert (
            cache.get_entry(
                model_id=model_a,
                prompt=prompts[i],
                **params[i],
            )
            == responses[i]
        )

    # Clear the entry with the newest creation time
    cache.clear(created_after=timestamp_0)

    # Check that entry n is not in the cache, but the others are
    assert cache.get_entry(model_id=model_a, prompt=prompts[-1], **params[-1]) is None

    for i in range(0, n_entries - 1):
        assert (
            cache.get_entry(model_id=model_a, prompt=prompts[i], **params[i])
            == responses[i]
        )

    time.sleep(1)
    timestamp_1 = now()
    time.sleep(1)

    # Access entries 1, n-2
    for i in range(1, n_entries - 1):
        assert (
            cache.get_entry(model_id=model_a, prompt=prompts[i], **params[i])
            == responses[i]
        )

    # Clear all entries accessed after timestamp_1
    cache.clear(accessed_after=timestamp_1, model_id=model_a)

    # Confirm only entry 0 is in the cache
    assert (
        cache.get_entry(model_id=model_a, prompt=prompts[0], **params[0])
        == responses[0]
    )

    for i in range(1, n_entries):
        assert cache.get_entry(model_id=model_a, prompt=prompts[i], **params[i]) is None

    # Clear the cache
    cache.clear(model_id=model_b)

    # Confirm that we still have entry 1 in the cache
    assert (
        cache.get_entry(model_id=model_a, prompt=prompts[0], **params[0])
        == responses[0]
    )
    assert cache.count_entries() == {model_a: 1}

    # Clear the cache again, but this time for model_a
    cache.clear(model_id=model_a)

    # Check that all entries are cleared
    for i in range(n_entries):
        assert cache.get_entry(model_id=model_a, prompt=prompts[i], **params[i]) is None
    assert cache.count_entries() == {}


def test_cache_access_times(cache: CacheDB) -> None:
    """
    Test cache access times.
    """

    # Constants: model identifiers
    model_a = "model a"

    # Constants: prompts
    prompt_a = "prompt a"
    prompt_b = "prompt b"

    # Constants: responses
    response_single = "test_response"

    # Add a new single response
    t0 = now()
    cache.add_entry(
        model_id=model_a,
        prompt=prompt_a,
        responses=response_single,
    )
    t1 = now()

    # Get the single response
    assert cache.get_entry(model_id=model_a, prompt=prompt_a) == [response_single]

    # Get the access time of the cache entry
    access_times = cache.get_earliest_access_times()
    print(access_times)
    print(t0)
    print(t1)
    assert len(access_times) == 1
    assert t0 <= access_times[model_a]
    assert access_times[model_a] <= t1

    # Access the cache entry again
    t2 = now()
    cache.get_entry(model_id=model_a, prompt=prompt_a)
    t3 = now()

    # Get the access time of the cache entry
    access_times = cache.get_earliest_access_times()
    assert len(access_times) == 1
    assert t2 <= access_times[model_a] <= t3

    # The latest access time should be the same as the earliest access time because
    # there is only one cache entry
    assert cache.get_latest_access_times() == access_times

    # Get the creation time of the cache entry, this should be the same as the original
    # access time
    creation_times = cache.get_earliest_creation_times()
    assert len(creation_times) == 1
    assert t0 <= creation_times[model_a] <= t1

    # The latest creation time should be the same as the earliest creation time because
    # there is only one cache entry
    assert cache.get_latest_creation_times() == creation_times

    # Add a new single response
    t4 = now()
    cache.add_entry(
        model_id=model_a,
        prompt=prompt_b,
        responses=response_single,
    )
    t5 = now()

    # The earliest access time should be the same as before
    access_times = cache.get_earliest_access_times()
    assert len(access_times) == 1
    assert t2 <= access_times[model_a] <= t3

    # The latest access time should be newer
    access_times = cache.get_latest_access_times()
    assert len(access_times) == 1
    assert t4 <= access_times[model_a] <= t5

    # The earliest creation time should be the same as before
    creation_times = cache.get_earliest_creation_times()
    assert len(creation_times) == 1
    assert t0 <= creation_times[model_a] <= t1

    # The latest creation time should be newer
    creation_times = cache.get_latest_creation_times()
    assert len(creation_times) == 1
    assert t4 <= creation_times[model_a] <= t5


def test_invalid_values(cache: CacheDB) -> None:
    # Attempt to add an entry with an invalid parameter type
    with pytest.raises(
        TypeError,
        match=(
            r"^Model parameters must be strings, integers, floats, or booleans, but "
            r"got parameter invalid_param=\['invalid'\]$"
        ),
    ):
        cache.add_entry(
            model_id="model",
            prompt="prompt",
            responses="response",
            # a list is not a valid parameter type
            invalid_param=["invalid"],  # type: ignore[arg-type]
        )

    # Confirm that the cache is still empty
    assert cache.count_entries() == {}

    # Attempt to access an entry with an invalid parameter type
    with pytest.raises(
        TypeError,
        match=(
            r"^Model parameters must be strings, integers, floats, or booleans, but "
            r"got parameter invalid_param=\['invalid'\]$"
        ),
    ):
        cache.get_entry(
            model_id="model",
            prompt="prompt",
            # a list is not a valid parameter type
            invalid_param=["invalid"],  # type: ignore[arg-type]
        )


def test_parse_iso_utc() -> None:
    """
    Test the parse_iso_utc function.
    """
    # noinspection PyProtectedMember
    from artkit.model.cache._cache import _parse_iso_utc as parse_iso_utc

    # Test the function
    assert parse_iso_utc("2021-01-01T00:00:00") == datetime(
        2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc
    )
    assert parse_iso_utc("2021-01-01T00:00:00Z") == datetime(
        2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc
    )

    # Test the function with an invalid input
    with pytest.raises(ValueError):
        parse_iso_utc("2021-13-01T00:00:00")


#
# Auxiliary functions
#


def now() -> datetime:
    # Get the current time in UTC
    return datetime.now(tz=timezone.utc).replace(microsecond=0)
