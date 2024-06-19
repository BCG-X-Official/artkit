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
Implementation of the Image class for I/O handling of Vision and Diffusion models.
"""

import base64
from typing import Any, overload

from pytools.api import inheritdoc
from pytools.expression import Expression, HasExpressionRepr
from pytools.expression.atomic import Id
from pytools.http import fetch_url

__all__ = [
    "Image",
]


#
# Classes
#


@inheritdoc(match="""[see superclass]""")
class Image(HasExpressionRepr):
    """Representation of an image for I/O handling of Vision and Diffusion models"""

    #: Binary data of image
    _data: bytes | None = None

    #: URL of image
    url: str | None = None

    @overload
    def __init__(self, *, data: bytes) -> None:
        """[see below]"""

    @overload
    def __init__(self, *, url: str) -> None:
        """[see below]"""

    def __init__(self, *, data: bytes | None = None, url: str | None = None) -> None:
        """
        :param data: the binary image data
        :param url: a URL to fetch the image from
        """
        if data is not None:
            if url is not None:
                raise ValueError("Exactly one of arg data and arg url must be provided")
            self._data = data
        elif url is not None:
            self.url = url
        else:
            raise ValueError("Either arg data or arg url must be provided")

    @property
    def data(self) -> bytes:
        """
        The binary image data.
        """
        data = self._data
        if data is None:
            # Fetch the image data from the URL
            self._data = data = self._fetch_image_from_url()

        return data

    def _fetch_image_from_url(self) -> bytes:
        """
        Download an image from URL and encodes it in binary format.

        :return: the binary image data
        """
        url = self.url
        assert url is not None, "URL must be set to fetch image from URL"

        # Send a GET request to the image URL
        return fetch_url(url)

    def to_b64(self) -> bytes:
        """
        Encode the image data into base64 format.

        :return: the base64-encoded image
        """
        return base64.b64encode(self.data)

    def to_b64_string(self) -> str:
        """
        Encode the image data as a base64 string.

        The main purpose of this method is to allow for JSON serialization.

        :return: a base64-encoded string representing the image
        """
        return base64.b64encode(self.data).decode("ascii")

    def show(
        self, width: int | None = None, height: int | None = None
    ) -> None:  # pragma: no cover
        """
        Display the image using IPython.

        :param width: the width of the image (optional)
        :param height: the height of the image (optional)
        """
        try:
            from IPython.display import Image, display

            display(Image(data=self.data, width=width, height=height))
        except ImportError as e:
            raise ImportError("IPython is required to display images") from e

    def __hash__(self) -> int:
        return hash((self.url, self.data))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Image)
            and self.url == other.url
            and self.data == other.data
        )

    def _repr_png_(self) -> bytes:
        """
        Render the image as a PNG for Jupyter notebooks.

        :return: the PNG data
        """
        return self.data

    def to_expression(self) -> Expression:
        """[see superclass]"""
        id = Id(type(self))
        if self.url is not None:
            return id(url=self.url)
        else:
            return id(data=self.data)
