import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from openai import RateLimitError

from artkit.model.util import RateLimitException
from artkit.model.vision import CachedVisionModel
from artkit.model.vision.base import VisionModel
from artkit.model.vision.openai import OpenAIVision
from artkit.util import Image

openai = pytest.importorskip("openai")


@pytest.mark.asyncio
async def test_vision_openai(
    image: Image,
    cached_vision_openai: VisionModel,
) -> None:
    messages = await cached_vision_openai.image_to_text(
        image=image, prompt="What color is this image? Answer in one word."
    )
    assert "black" in messages[0].lower()


@pytest.mark.asyncio
async def test_openai_retry(image: Image, caplog: pytest.LogCaptureFixture) -> None:
    with patch("artkit.model.vision.openai._openai.AsyncOpenAI") as MockClientSession:
        api_key_env = "TEST"
        os.environ[api_key_env] = "test"
        openai_chat = OpenAIVision(
            model_id="gpt-4-turbo",
            api_key_env=api_key_env,
            initial_delay=0.1,
            exponential_base=1.5,
            max_retries=2,
        )

        # Response mock
        response = Mock()
        response.status_code = 429

        MockClientSession.return_value.chat.completions.create.side_effect = (
            RateLimitError(
                message="Rate Limit exceeded",
                response=response,
                body=Mock(),
            )
        )

        with pytest.raises(RateLimitException):
            await openai_chat.image_to_text(image=image, message="Any image")
        assert MockClientSession.return_value.chat.completions.create.call_count == 2

    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("Rate limit exceeded")
            ]
        )
        == openai_chat.max_retries
    )


@pytest.fixture
def cached_vision_openai(data_path: Path) -> Iterator[VisionModel]:
    database = data_path / "_copy_vision_openai.db"
    shutil.copyfile(data_path / "vision_openai.db", database)

    cached = CachedVisionModel(
        model=OpenAIVision(model_id="gpt-4-turbo"), database=database
    )

    yield cached

    os.remove(database)
