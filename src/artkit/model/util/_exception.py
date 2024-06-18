"""
This file contains custom exception classes for the models.
"""

__all__ = [
    "RateLimitException",
]


class RateLimitException(Exception):
    """
    An exception raised when an error occurs while communicating with a model.
    """

    pass
