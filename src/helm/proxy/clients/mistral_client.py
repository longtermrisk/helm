# mypy: check_untyped_defs = False
import copy
import os
import time
from datetime import datetime

import json

import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional, cast
from typing import Union

import httpx


from helm.benchmark.model_metadata_registry import is_vlm
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import (
    wrap_request_time,
    Request,
    RequestResult,
    Sequence,
    Token,
)
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.proxy.tokenizers.tokenizer import Tokenizer
from .client import (
    CachingClient,
    truncate_sequence,
    generate_uid_for_multimodal_prompt,
)
from ...common.clr_constants import (
    log_api_request,
    MISTRAL_CLIENT_LOG_FILE,
    USE_SINGLE_STEP_SG_IMPLEMENTATION,
    USE_THREE_STEPS_SG_IMPLEMENTATION,
    USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT,
)

try:
    import mistralai
    import mistralai.exceptions
    from mistralai.client import MistralClient

except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["mistralai"])


class MistralAIClient(CachingClient):
    # END_OF_TEXT: str = "<|endoftext|>"
    MISTRALAI_CLIENT = None

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        # self.org_id: Optional[str] = org_id
        self.api_key: Optional[str] = (
            api_key if api_key is not None else os.getenv("MISTRAL_API_KEY")
        )
        self.client = MistralClient(api_key=self.api_key, max_retries=0)

    def _is_chat_model_engine(self, model_engine: str):
        return True

    def _get_cache_key(self, raw_request, request):
        cache_key = CachingClient.make_cache_key(
            {
                "USE_SINGLE_STEP_SG_IMPLEMENTATION": (
                    USE_SINGLE_STEP_SG_IMPLEMENTATION
                ),
                "USE_THREE_STEPS_SG_IMPLEMENTATION": (
                    USE_THREE_STEPS_SG_IMPLEMENTATION
                ),
                "USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT": (
                    USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT
                ),
                "caching_index": request.caching_index,
                "debugging_index": 0,
                **raw_request,
            },
            request,
        )
        if is_vlm(request.model):
            assert request.multimodal_prompt is not None
            prompt_key: str = generate_uid_for_multimodal_prompt(
                request.multimodal_prompt
            )
            cache_key = {**cache_key, "multimodal_prompt": prompt_key}
            del cache_key["messages"]
        return cache_key

    def make_request(self, request: Request) -> RequestResult:
        if self.api_key is None:
            raise ValueError("MistralAI API key is required")

        raw_request: Dict[str, Any]
        if request.embedding:
            raise NotImplementedError()
        elif self._is_chat_model_engine(request.model_engine):
            messages: Optional[
                List[Dict[str, Union[str, Any]]]
            ] = request.messages
            if request.messages and len(request.messages) > 1:
                # Checks that all messages have a role and some content
                for message in request.messages:
                    if not message.get("role") or not message.get("content"):
                        raise ValueError(
                            "All messages must have a role and content"
                        )
                # Checks that the last role is "user"
                if request.messages[-1]["role"] != "user":
                    raise ValueError("Last message must have role 'user'")
                if request.prompt != "":
                    hlog(
                        "WARNING: Since message is set, prompt will be ignored"
                    )
            else:
                content: Union[str, List[Union[str, Any]]]
                if request.multimodal_prompt is not None:
                    raise NotImplementedError()
                else:
                    content = request.prompt

                messages = [{"role": "user", "content": content}]

            raw_request = {
                "model": request.model_engine,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.num_completions,
                # "stop": (request.stop_sequences or None),
                "max_tokens": request.max_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
            }

        else:
            raise NotImplementedError()

        if request.embedding:
            raise NotImplementedError()

        elif self._is_chat_model_engine(request.model_engine):

            def do_it():
                adapted_raw_request = MistralAIClient.adapt_kwargs(
                    **raw_request
                )
                start_time = time.time()
                response = self.client.chat(**adapted_raw_request)
                end_time = time.time()
                time_to_wait = 0.20 - (end_time - start_time)
                if time_to_wait > 0:
                    time.sleep(
                        time_to_wait
                    )  # Rate limit at 5 requests per second
                return MistralAIClient.to_openai_response_format(response)

        else:
            raise NotImplementedError()

        # if raw_request["max_tokens"] == 0 and self._is_chat_model_engine(
        #     request.model_engine
        # ):
        #     print(
        #         "WARNING: You are requesting an empty completion while this is"
        #         " not supported by the OpenAI chat API."
        #     )
        #     print(
        #         "WARNING suite 1: You are likely trying to use a benchmark that"
        #         " requires logprobs while the OpenAI chat API doesn't provide"
        #         " that"
        #     )
        #     print("WARNING suite 2: raw_request =", raw_request)

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(
                cache_key, wrap_request_time(do_it)
            )
        except (
            httpx.ReadTimeout,
            mistralai.exceptions.MistralAPIStatusException,
        ) as e:
            error: str = (
                f"MistralAI error: {e}, traceback: {traceback.format_exc()}"
            )
            return RequestResult(
                success=False,
                cached=False,
                error=error,
                completions=[],
                embedding=[],
            )

        # If the user is requesting completions instead of an embedding, then `completions`
        # needs to be populated, and `embedding` should be an empty list and vice-versa.
        embedding: List[float] = []
        completions: List[Sequence] = []
        tokens: List[Token]
        if request.embedding:
            raise NotImplementedError()
        elif self._is_chat_model_engine(request.model_engine):
            for raw_completion in response["choices"]:
                # The OpenAI chat completion API doesn't support echo.
                # If `echo_prompt` is true, combine the prompt and completion.
                raw_completion_content = raw_completion["message"]["content"]
                text: str = (
                    request.prompt + raw_completion_content
                    if request.echo_prompt
                    else raw_completion_content
                )
                # The MistralAI chat completion API doesn't return us tokens or logprobs, so we tokenize ourselves.
                tokenization_result: TokenizationRequestResult = (
                    self.tokenizer.tokenize(
                        TokenizationRequest(text, tokenizer=self.tokenizer_name)
                    )
                )
                # Log probs are not currently not supported by the MistralAI chat completion API, so set to 0 for now.
                tokens = [
                    Token(text=cast(str, raw_token), logprob=0, top_logprobs={})
                    for raw_token in tokenization_result.raw_tokens
                ]
                completion = Sequence(
                    text=text,
                    logprob=0,  # MistralAI does not provide logprobs
                    tokens=tokens,
                    finish_reason={
                        "reason": raw_completion.get("finish_reason", "NAN")
                    },
                )
                completions.append(
                    truncate_sequence(completion, request)
                )  # Truncate the text by stop sequences
        else:
            raise NotImplementedError()

        log_api_request(
            MISTRAL_CLIENT_LOG_FILE,
            request=request,
            raw_request=raw_request,
            response=response,
        )

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=embedding,
        )

    @staticmethod
    def adapt_kwargs(**kwargs):
        from mistralai.models.chat_completion import (
            ChatMessage as mistralai_ChatMessage,
        )

        kwargs = copy.deepcopy(kwargs)
        if "n" in kwargs.keys():
            assert kwargs["n"] == 1, f"n='{kwargs['n']}' not supported"
            kwargs.pop("n")

        presence_penalty = kwargs.pop("presence_penalty", None)
        frequency_penalty = kwargs.pop("frequency_penalty", None)
        stop_sequences = kwargs.pop("stop", None)
        assert (
            presence_penalty is None or presence_penalty == 0
        ), f"presence_penalty='{presence_penalty}' not supported"
        assert (
            frequency_penalty is None or frequency_penalty == 0
        ), f"frequency_penalty='{frequency_penalty}' not supported"
        assert stop_sequences is None, f"stop='{stop_sequences}' not supported"

        kwargs["messages"] = [
            mistralai_ChatMessage(
                role=message["role"], content=message["content"]
            )
            for message in kwargs["messages"]
        ]
        return kwargs

    @staticmethod
    def to_openai_response_format(response):
        openai_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    },
                }
            ],
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": response.model,
            #         "system_fingerprint": chat_object.system_fingerprint,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        return openai_response
