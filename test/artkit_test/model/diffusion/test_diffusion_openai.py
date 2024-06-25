import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from openai import RateLimitError

from artkit.model.diffusion import CachedDiffusionModel
from artkit.model.diffusion.base import DiffusionModel
from artkit.model.diffusion.openai import OpenAIDiffusion
from artkit.model.util import RateLimitException

openai = pytest.importorskip("openai")


@pytest.mark.asyncio
async def test_diffusion_openai(
    cached_openai: DiffusionModel,
) -> None:
    img = await cached_openai.text_to_image(text="Can you draw me a picture?")
    assert img[0].to_b64_string() == "MA=="


@pytest.mark.asyncio
async def test_openai_retry(caplog: pytest.LogCaptureFixture) -> None:
    with patch(
        "artkit.model.diffusion.openai._openai.AsyncOpenAI"
    ) as MockClientSession:
        api_key_env = "TEST"
        os.environ[api_key_env] = "test"
        openai_chat = OpenAIDiffusion(
            model_id="dall-e-3",
            api_key_env=api_key_env,
            initial_delay=0.1,
            exponential_base=1.5,
            max_retries=2,
        )

        # Response mock
        response = Mock()
        response.status_code = 429

        MockClientSession.return_value.images.generate.side_effect = RateLimitError(
            message="Rate Limit exceeded",
            response=response,
            body=Mock(),
        )

        with pytest.raises(RateLimitException):
            await openai_chat.text_to_image(text="Can you draw me a picture?")
        assert (
            MockClientSession.return_value.images.generate.call_count
            == openai_chat.max_retries
        )

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
def cached_openai(data_path: Path) -> Iterator[DiffusionModel]:
    database = data_path / "_copy_diffusion_openai.db"
    shutil.copyfile(data_path / "diffusion_openai.db", database)

    yield CachedDiffusionModel(
        model=OpenAIDiffusion(model_id="dall-e-3"),
        database=database,
    )

    os.remove(database)
