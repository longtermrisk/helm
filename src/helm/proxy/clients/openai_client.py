# mypy: check_untyped_defs = False
import os
from datetime import datetime

import json

import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional, cast
from typing import Union

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
    OPENAI_CLIENT_LOG_FILE,
    USE_SINGLE_STEP_SG_IMPLEMENTATION,
    USE_THREE_STEPS_SG_IMPLEMENTATION,
    USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT,
)

try:
    import openai
    import openai.types.chat
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


# ORIGINAL_COMPLETION_ATTRIBUTES = (
#     openai.api_resources.completion.Completion.__bases__
# )


class OpenAIClient(CachingClient):
    END_OF_TEXT: str = "<|endoftext|>"
    OPENAI_CLIENT = None

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
        self.org_id: Optional[str] = org_id
        self.api_key: Optional[str] = (
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        )
        self.api_base: str = "https://api.openai.com/v1"

    def _is_chat_model_engine(self, model_engine: str):
        return "gpt-3.5" in model_engine or "gpt-4" in model_engine

    def _set_access_info(self):
        # Following https://beta.openai.com/docs/api-reference/authentication
        # `organization` can be set to None.
        openai.organization = self.org_id
        openai.api_key = self.api_key
        openai.api_base = self.api_base

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
            raise ValueError("OpenAI API key is required")

        raw_request: Dict[str, Any]
        if request.embedding:
            raw_request = {
                "input": request.prompt,
                "engine": request.model_engine,
            }
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
                # Convert prompt into a single message
                # For now, put the whole prompt in a single user message, and expect the response
                # to be returned in a single assistant message.
                # TODO: Support ChatML for creating multiple messages with different roles.
                # See: https://github.com/openai/openai-python/blob/main/chatml.md

                # Content can either be text or a list of multimodal content made up of text and images:
                # https://platform.openai.com/docs/guides/vision
                content: Union[str, List[Union[str, Any]]]
                if request.multimodal_prompt is not None:
                    content = []
                    for media_object in request.multimodal_prompt.media_objects:
                        if (
                            media_object.is_type("image")
                            and media_object.location
                        ):
                            from helm.common.images_utils import encode_base64

                            base64_image: str = encode_base64(
                                media_object.location
                            )
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                }
                            )
                        elif media_object.is_type(TEXT_TYPE):
                            if media_object.text is None:
                                raise ValueError(
                                    "MediaObject of text type has missing text"
                                    " field value"
                                )
                            content.append(
                                {
                                    "type": media_object.type,
                                    "text": media_object.text,
                                }
                            )
                        else:
                            raise ValueError(
                                "Unrecognized MediaObject type"
                                f" {media_object.type}"
                            )

                else:
                    content = request.prompt

                messages = [{"role": "user", "content": content}]

            raw_request = {
                "model": request.model_engine,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.num_completions,
                "stop": (
                    request.stop_sequences or None
                ),  # API doesn't like empty list
                # Note: Chat models may require adding an extra token to max_tokens
                # for the internal special role token.
                "max_tokens": request.max_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
            }

            # OpenAI's vision API doesn't allow None values for stop.
            # Fails with "body -> stop: none is not an allowed value" if None is passed.
            if is_vlm(request.model) and raw_request["stop"] is None:
                raw_request.pop("stop")
        else:
            raw_request = {
                "engine": request.model_engine,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "n": request.num_completions,
                "max_tokens": request.max_tokens,
                "best_of": request.top_k_per_token,
                "logprobs": request.top_k_per_token,
                "stop": (
                    request.stop_sequences or None
                ),  # API doesn't like empty list
                "top_p": request.top_p,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "echo": request.echo_prompt,
            }

            # OpenAI doesn't let you ask for more completions than the number of
            # per-token candidates.
            raw_request["best_of"] = max(
                raw_request["best_of"], raw_request["n"]
            )
            raw_request["logprobs"] = max(
                raw_request["logprobs"], raw_request["n"]
            )

        if request.embedding:

            def do_it():
                self._set_access_info()
                raise NotImplementedError()
                # return openai.Embedding.create(**raw_request)

        elif self._is_chat_model_engine(request.model_engine):

            def do_it():
                self._set_access_info()
                if self.OPENAI_CLIENT is None:
                    from openai import OpenAI

                    self.OPENAI_CLIENT = OpenAI(
                        api_key=open(
                            "/Users/maximeriche/Dev/surrogate_goals_demo/openai_key.txt",
                            "r",
                        )
                        .read()
                        .strip()
                    )
                chat_object = self.OPENAI_CLIENT.chat.completions.create(
                    **raw_request
                )
                # assert 0

                return self.new_to_old_openai_response_format(chat_object)

                # return openai.ChatCompletion.create(**raw_request)

        else:

            def do_it():
                self._set_access_info()
                raise NotImplementedError()
                # openai.api_resources.completion.Completion.__bases__ = (
                #     ORIGINAL_COMPLETION_ATTRIBUTES
                # )
                # return openai.Completion.create(**raw_request)

        if raw_request["max_tokens"] == 0 and self._is_chat_model_engine(
            request.model_engine
        ):
            print(
                "WARNING: You are requesting an empty completion while this is"
                " not supported by the OpenAI chat API."
            )
            print(
                "WARNING suite 1: You are likely trying to use a benchmark that"
                " requires logprobs while the OpenAI chat API doesn't provide"
                " that"
            )
            print("WARNING suite 2: raw_request =", raw_request)

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(
                cache_key, wrap_request_time(do_it)
            )
        except openai.OpenAIError as e:
            error: str = (
                f"OpenAI error: {e}, traceback: {traceback.format_exc()}"
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
            # If the user is requesting an embedding instead of completion
            # then completions would be left as an empty list. The embedding needs to be set.
            embedding = response["data"][0]["embedding"]
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
                # The OpenAI chat completion API doesn't return us tokens or logprobs, so we tokenize ourselves.
                tokenization_result: TokenizationRequestResult = (
                    self.tokenizer.tokenize(
                        TokenizationRequest(text, tokenizer=self.tokenizer_name)
                    )
                )
                # Log probs are not currently not supported by the OpenAI chat completion API, so set to 0 for now.
                tokens = [
                    Token(text=cast(str, raw_token), logprob=0, top_logprobs={})
                    for raw_token in tokenization_result.raw_tokens
                ]
                completion = Sequence(
                    text=text,
                    logprob=0,  # OpenAI does not provide logprobs
                    tokens=tokens,
                    finish_reason={"reason": raw_completion["finish_reason"]},
                )
                completions.append(
                    truncate_sequence(completion, request)
                )  # Truncate the text by stop sequences
        else:
            for raw_completion in response["choices"]:
                sequence_logprob = 0
                tokens = []

                raw_data = raw_completion["logprobs"]
                for text, logprob, top_logprobs in zip(
                    raw_data["tokens"],
                    raw_data["token_logprobs"],
                    raw_data["top_logprobs"],
                ):
                    tokens.append(
                        Token(
                            text=text,
                            logprob=logprob or 0,
                            top_logprobs=dict(top_logprobs or {}),
                        )
                    )
                    sequence_logprob += logprob or 0
                completion = Sequence(
                    text=raw_completion["text"],
                    logprob=sequence_logprob,
                    tokens=tokens,
                    finish_reason={"reason": raw_completion["finish_reason"]},
                )
                # OpenAI sends us back tokens past the end of text token,
                # so we need to manually truncate the list of tokens.
                # TODO: filed an issue with their support to check what the expected behavior here is.
                completion = truncate_sequence(
                    completion,
                    replace(
                        request,
                        stop_sequences=request.stop_sequences
                        + [OpenAIClient.END_OF_TEXT],
                    ),
                )
                completions.append(completion)

        log_api_request(
            OPENAI_CLIENT_LOG_FILE,
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

    def new_to_old_openai_response_format(
        self, chat_object: openai.types.chat.ChatCompletion
    ):
        openai_response = {
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": "assistant",
                        "content": choice.message.content,
                    },
                    "logprobs": choice.logprobs,
                    "finish_reason": choice.finish_reason,
                }
                for choice in chat_object.choices
            ],
            "id": chat_object.id,
            "object": "chat.completion",
            "created": chat_object.created,
            "model": chat_object.model,
            "system_fingerprint": chat_object.system_fingerprint,
            "usage": {
                "prompt_tokens": chat_object.usage.prompt_tokens,
                "completion_tokens": chat_object.usage.completion_tokens,
                "total_tokens": chat_object.usage.total_tokens,
            },
        }
        return openai_response
