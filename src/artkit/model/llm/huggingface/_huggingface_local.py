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
Huggingface LLM systems that run locally.
"""

import logging
from typing import Any, cast

from pytools.api import MissingClassMeta, inheritdoc

from .base import (
    HuggingfaceChatConnectorMixin,
    HuggingfaceCompletionConnectorMixin,
    HuggingfaceLocalConnectorMixin,
)

log = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover

    class AutoModelForCausalLM(  # type: ignore
        metaclass=MissingClassMeta, module="transformers"
    ):
        """Placeholder class for missing ``AutoModelForCausalLM`` class."""

    class AutoTokenizer(  # type: ignore
        metaclass=MissingClassMeta,
        module="transformers",
    ):
        """Placeholder class for missing ``AutoTokenizer`` class."""


__all__ = [
    "HuggingfaceCompletionLocal",
    "HuggingfaceChatLocal",
]


#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class HuggingfaceCompletionLocal(
    HuggingfaceLocalConnectorMixin,
    HuggingfaceCompletionConnectorMixin[AutoModelForCausalLM],
):
    """
    Huggingface completion model for local inference.
    """

    async def _make_completion(
        self, prompt: str, **model_params: dict[str, Any]
    ) -> str:
        """[see superclass]"""
        tokenizer = self.get_tokenizer()
        client = self.get_client()

        # noinspection PyCallingNonCallable
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        if self.use_cuda:
            input_ids = input_ids.to("cuda")

        # noinspection PyUnresolvedReferences
        outputs = client.generate(
            input_ids,  # We set `do_sample=True` to allow for randomness in the output
            do_sample=True,
            **{**self.get_model_params(), **model_params},
        )

        # noinspection PyUnresolvedReferences
        return cast(str, tokenizer.decode(outputs[0], skip_special_tokens=True))


@inheritdoc(match="""[see superclass]""")
class HuggingfaceChatLocal(
    HuggingfaceLocalConnectorMixin, HuggingfaceChatConnectorMixin[AutoModelForCausalLM]
):
    """
    Huggingface model base local chat.
    """

    async def _get_client_response(
        self,
        messages: list[dict[str, str]],
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""
        client: AutoModelForCausalLM = self.get_client()
        # noinspection PyUnresolvedReferences
        tokenizer = self.get_tokenizer()

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            **self.chat_template_params,
        ).to(client.device)

        # noinspection PyUnresolvedReferences
        outputs = client.generate(
            input_ids,
            **{**self.get_model_params(), **model_params},
        )

        response = outputs[0][input_ids.shape[-1] :]
        # noinspection PyUnresolvedReferences
        return [tokenizer.decode(response, skip_special_tokens=True)]
