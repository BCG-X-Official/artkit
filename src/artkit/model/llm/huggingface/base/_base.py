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
Huggingface LLM systems.
"""

import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import AsyncExitStack
from typing import Any, Generic, TypeVar, final

from pytools.api import (
    MissingClassMeta,
    appenddoc,
    inheritdoc,
    missing_function,
    subsdoc,
)

from ....base import ConnectorMixin
from ...base import ChatModelConnector, CompletionModelConnector
from ...history import ChatHistory, ChatMessage, SystemPrompt, UserMessage

log = logging.getLogger(__name__)

__all__ = [
    "HuggingfaceChatConnectorMixin",
    "HuggingfaceCompletionConnectorMixin",
    "HuggingfaceConnectorMixin",
    "HuggingfaceLocalConnectorMixin",
    "HuggingfaceRemoteConnectorMixin",
]

try:
    # noinspection PyUnresolvedReferences
    from huggingface_hub import AsyncInferenceClient
    from torch.cuda import is_available

    # noinspection PyUnresolvedReferences
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover

    is_available = missing_function(name="is_available", module="torch.cuda")

    class AsyncInferenceClient(  # type: ignore
        metaclass=MissingClassMeta, module="huggingface_hub"
    ):
        """Placeholder class for missing ``AsyncInferenceClient`` class."""

    class AutoModelForCausalLM(  # type: ignore
        metaclass=MissingClassMeta, module="transformers"
    ):
        """Placeholder class for missing ``AutoModelForCausalLM`` class."""

    class AutoTokenizer(metaclass=MissingClassMeta, module="transformers"):  # type: ignore
        """Placeholder class for missing ``AutoTokenizer`` class."""

    class PreTrainedTokenizerBase(metaclass=MissingClassMeta, module="transformers"):
        """Placeholder class for missing ``PreTrainedTokenizerBase`` class."""


#
# Type variables
#

T_Client = TypeVar("T_Client")

#
# Class declarations
#


@inheritdoc(match="""[see superclass]""")
class HuggingfaceConnectorMixin(
    ConnectorMixin[T_Client], Generic[T_Client], metaclass=ABCMeta
):
    """
    Base class for Huggingface LLMs.
    """

    #: The tokenizer associated with the connector instance.
    __tokenizer: AutoTokenizer | None = None

    @classmethod
    def get_default_api_key_env(cls) -> str:
        """[see superclass]"""
        return "HF_TOKEN"

    @final
    def get_tokenizer(self) -> AutoTokenizer:  # pragma: no cover
        """
        The tokenizer used by this connector.

        Gets loaded on first access.

        :return: the tokenizer
        """
        if self.__tokenizer is None:
            self.__tokenizer = self._make_tokenizer()
        return self.__tokenizer

    def _make_tokenizer(self) -> AutoTokenizer:  # pragma: no cover
        return AutoTokenizer.from_pretrained(self.model_id)


class HuggingfaceLocalConnectorMixin(HuggingfaceConnectorMixin[AutoModelForCausalLM]):
    """
    Base class for Hugging Face local models.
    """

    #: If ``True``, use CUDA; if ``False``, use CPU
    use_cuda: bool

    @subsdoc(
        # Remove the `model_params` parameter from the superclass
        pattern=r":param model_params:.*\n(?!:raises)",
        replacement="",
    )
    @subsdoc(
        # Remove parameters from the superclass
        pattern=r":param initial_delay:(?:.|\n)+:param max_retries:.*\n",
        replacement="",
    )
    @appenddoc(to=HuggingfaceConnectorMixin.__init__)
    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str | None = None,
        use_cuda: bool = False,
        **model_params: Any,
    ) -> None:
        """
        :param use_cuda: if ``True``, use CUDA; if ``False``, use CPU
        :param model_params: additional parameters for the model
        :raises RuntimeError: if CUDA is requested but not available
        """
        super().__init__(
            model_id=model_id,
            api_key_env=api_key_env,
            model_params=model_params,
        )

        # test if cuda is available
        if use_cuda and not is_available():  # pragma: no cover
            raise RuntimeError("CUDA requested but not available.")

        self.use_cuda = use_cuda

    def _make_client(self) -> AutoModelForCausalLM:  # pragma: no cover
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.use_cuda:
            # We set the device to use cuda
            model.to("cuda")
        return model


class HuggingfaceRemoteConnectorMixin(HuggingfaceConnectorMixin[AsyncInferenceClient]):
    """
    Base class for Huggingface remote models.
    """

    def _make_client(self) -> AsyncInferenceClient:
        return AsyncInferenceClient(model=self.model_id, token=self.get_api_key())


@inheritdoc(match="""[see superclass]""")
class HuggingfaceCompletionConnectorMixin(
    HuggingfaceConnectorMixin[T_Client],
    CompletionModelConnector[T_Client],
    Generic[T_Client],
    metaclass=ABCMeta,
):
    """
    Base class for Huggingface completion model connectors.
    """

    async def get_completion(
        self, prompt: str, **model_params: dict[str, Any]
    ) -> str:  # pragma: no cover
        """[see superclass]"""

        response = await self._make_completion(prompt, **model_params)

        if response.startswith(prompt):
            # remove the prompt from the response
            return response[len(prompt) :]
        else:
            return response

    @abstractmethod
    async def _make_completion(
        self, prompt: str, **model_params: dict[str, Any]
    ) -> str:
        """
        Generate a completion for the given prompt.

        :param prompt: the prompt to generate a completion for
        :param model_params: additional parameters for the model
        :return: the generated completion, either including or excluding the prompt
        :raises RateLimitException: if the client returns a rate limit error
        """


@inheritdoc(match="""[see superclass]""")
class HuggingfaceChatConnectorMixin(
    HuggingfaceConnectorMixin[T_Client],
    ChatModelConnector[T_Client],
    Generic[T_Client],
    metaclass=ABCMeta,
):
    """
    Base class for Huggingface chat systems.

    :raises ValueError: if the tokenizer for the given model id does not support
        chat templates and no custom chat template was provided
    """

    #: The chat template to use for the model; ``None`` if using the default template.
    chat_template: str | None

    #: Additional chat template parameters.
    chat_template_params: Mapping[str, Any]

    @subsdoc(
        # The pattern matches the row defining model_params, and move it to the end
        # of the docstring.
        pattern=r"(:param model_params: .*\n)((:?.|\n)*\S)(\n|\s)*",
        replacement=r"\2\1",
    )
    @appenddoc(to=ChatModelConnector.__init__)
    def __init__(
        self,
        *,
        model_id: str,
        api_key_env: str = "HF_TOKEN",
        initial_delay: float | None = None,
        exponential_base: float | None = None,
        jitter: bool | None = None,
        max_retries: int | None = None,
        system_prompt: str | None = None,
        chat_template: str | None = None,
        chat_template_params: Mapping[str, Any] | None = None,
        **model_params: Any,
    ) -> None:
        """
        :param chat_template: an optional jinja template which will be used to
            format the input to the model based on hugging face implementation
            for more information see:\
                https://huggingface.co/docs/transformers/main/en/chat_templating
        :param chat_template_params: additional arguments to pass while formatting the
            input this allows reusing the same template for multiple models by
            customizing the template dynamically (optional)
        """

        super().__init__(
            model_id=model_id,
            api_key_env=api_key_env,
            initial_delay=initial_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            max_retries=max_retries,
            system_prompt=system_prompt,
            **model_params,
        )

        self.chat_template = chat_template
        self.chat_template_params = (
            chat_template_params if chat_template_params is not None else {}
        )

    def _make_tokenizer(self) -> AutoTokenizer:  # pragma: no cover
        model_id = self.model_id

        # Setup chat template
        tokenizer = super()._make_tokenizer()
        if self.chat_template is not None:
            if tokenizer.chat_template is not None:
                log.warning(
                    f"Tokenizer ({model_id}) natively supports chat templates, but "
                    f"will be overridden by the given custom chat template."
                )
            tokenizer.chat_template = self.chat_template
        elif tokenizer.chat_template is None:
            raise ValueError(
                f"Tokenizer {model_id !r} does not natively support chat templates; "
                f"need to specify custom chat template in arg chat_template"
            )
        return tokenizer

    def get_model_params(self) -> Mapping[str, Any]:
        """[see superclass]"""
        additional_params: dict[str, Any] = {}
        if self.chat_template_params:
            additional_params["chat_template_params"] = self.chat_template_params
        if self.chat_template is not None:
            additional_params["chat_template"] = self.chat_template
        if additional_params:
            return {**super().get_model_params(), **additional_params}
        else:
            return super().get_model_params()

    async def get_response(
        self,
        message: str,
        *,
        history: ChatHistory | None = None,
        **model_params: dict[str, Any],
    ) -> list[str]:
        """[see superclass]"""

        async with AsyncExitStack():

            reformatted_messages = [
                {"role": message.role, "content": message.text}
                for message in self._get_messages(message, history=history)
            ]

            response = await self._get_client_response(
                messages=reformatted_messages, **model_params
            )

        return response

    @abstractmethod
    async def _get_client_response(
        self, messages: list[dict[str, str]], **model_params: dict[str, Any]
    ) -> list[str]:
        """
        Get the response from the client.

        :param messages: the messages (including conversation history)
          to send to the client
        :param model_params: additional parameters for the model

        :return: the generated response from the client
        :raises RateLimitException: if the client returns a rate limit error
        """
        raise NotImplementedError()

    def _get_messages(
        self, message: str, *, history: ChatHistory | None
    ) -> Iterator[ChatMessage]:
        # Format the user prompt and system prompt (if defined)
        # as a list of chat messages for the Huggingface API.

        if self.system_prompt:
            yield SystemPrompt(self.system_prompt)

        if history is not None:
            yield from history.messages

        yield UserMessage(message)
