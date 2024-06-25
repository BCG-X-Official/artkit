import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from google.api_core.exceptions import TooManyRequests

from artkit.model.util import RateLimitException
from artkit.model.vision import CachedVisionModel
from artkit.model.vision.base import VisionModel
from artkit.model.vision.gemini import GeminiVision
from artkit.util import Image

_ = pytest.importorskip("google.generativeai")


@pytest.mark.asyncio
async def test_vision_gemini(
    image: Image,
    cached_vision_gemini: VisionModel,
) -> None:
    messages = await cached_vision_gemini.image_to_text(
        image=image, prompt="What color is this image? Answer in one word."
    )
    assert "black" in messages[0].lower()


@pytest.mark.asyncio
async def test_gemini_retry(
    gemini_vision: GeminiVision, image: Image, caplog: pytest.LogCaptureFixture
) -> None:
    with patch("artkit.model.vision.gemini._gemini.GeminiVision") as MockClientSession:
        # Response mock
        response = Mock()
        response.status_code = 429

        MockClientSession.return_value.chat.completions.create.side_effect = (
            TooManyRequests("Rate limit error")
        )

        with pytest.raises(RateLimitException):
            await gemini_vision.image_to_text(image=image, prompt="Any image")
        assert MockClientSession.return_value.chat.completions.create.call_count == 2

    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == gemini_vision.max_retries
    )


@pytest.fixture
def gemini_vision() -> GeminiVision:
    api_key_env = "TEST"
    os.environ[api_key_env] = "test"

    return GeminiVision(
        model_id="models/gemini-pro",
        api_key_env=api_key_env,
        max_output_tokens=10,
        max_retries=2,
        initial_delay=0.1,
        exponential_base=1.5,
    )


@pytest.fixture
def cached_vision_gemini(
    data_path: Path, gemini_vision: GeminiVision
) -> Iterator[VisionModel]:
    database = data_path / "_copy_gemini.db"
    shutil.copyfile(data_path / "gemini.db", database)

    yield CachedVisionModel(model=gemini_vision, database=database)

    os.remove(database)
