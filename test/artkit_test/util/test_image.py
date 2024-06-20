"""
Tests for the Image class
"""

from __future__ import annotations

import logging

import pytest

from artkit.util import Image

log = logging.getLogger(__name__)


def test_image_constructor() -> None:
    # Create an image from a URL

    image_from_url = Image(url="https://media-publications.bcg.com/BCG_MONOGRAM.png")
    assert image_from_url.url == "https://media-publications.bcg.com/BCG_MONOGRAM.png"
    assert image_from_url.to_b64_string().startswith("iVBO")
    assert image_from_url.to_b64().startswith(b"iVBO")
    assert image_from_url.data.startswith(b"\x89PNG")

    # Create an image from binary data

    image_from_data = Image(data=image_from_url.data)
    assert image_from_data.data == image_from_url.data
    assert image_from_data.to_b64_string() == image_from_url.to_b64_string()
    assert image_from_data.to_b64() == image_from_url.to_b64()

    # Create an image from a URL and binary data
    with pytest.raises(
        ValueError, match=r"^Exactly one of arg data and arg url must be provided$"
    ):
        Image(  # type: ignore[call-overload]
            url="https://media-publications.bcg.com/BCG_MONOGRAM.png", data=b"aaa"
        )

    # Create an image without a URL or binary data
    with pytest.raises(
        ValueError, match=r"^Either arg data or arg url must be provided$"
    ):
        Image()  # type: ignore[call-overload]


def test_image(image: Image) -> None:
    assert image._repr_png_() == image.data
    assert str(image) == f"Image(\n    data={image.data!r}\n)"

    image_with_url = Image(url="abc")
    # We set the data field so that the image does not attempt to fetch the URL
    image_with_url._data = image.data

    assert str(image_with_url) == "Image(url='abc')"


def test_image_comparison(image: Image) -> None:

    image_copy = Image(data=image.data)
    assert image == image_copy
    assert hash(image) == hash(image_copy)

    # Set the URL field so that both image objects are different
    image_copy.url = "abc"

    assert image != image_copy
    assert hash(image) != hash(image_copy)
