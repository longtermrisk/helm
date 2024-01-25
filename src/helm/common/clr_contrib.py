import copy
import os
from dataclasses import replace

import tiktoken
from anthropic import HUMAN_PROMPT, AI_PROMPT
from typing import List, Tuple

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.executor import Executor
from helm.common.request import RequestResult

import surrogate_goal_demo.analysis.utils.multi_step_SG_implementation as sg_demo
from helm.common.clr_constants import (
    USE_SINGLE_STEP_SG_IMPLEMENTATION,
    USE_THREE_STEPS_SG_IMPLEMENTATION,
    log_api_request,
    pick_right_log_file,
)
from surrogate_goal_demo.shared.external_loading_prompts import (
    load_single_step_sg_implementation_prompt,
    load_three_steps_sg_implementation_prompts,
    THREE_STEPS_SG_IMPLEMENTATION_VERSION_TO_USE,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


SINGLE_STEP_PROMPT = (
    load_single_step_sg_implementation_prompt()
    if USE_SINGLE_STEP_SG_IMPLEMENTATION
    else None
)
MULTI_STEP_PROMPT_STEP_1, MULTI_STEP_PROMPT_STEP_2 = (
    load_three_steps_sg_implementation_prompts()
    if USE_THREE_STEPS_SG_IMPLEMENTATION
    else (None, None)
)


class MultiStepExecutor(Executor):
    def process(self, state: RequestState) -> RequestState:
        self.tokenizer_service = None
        self.adapter = None

        initial_request = copy.deepcopy(state.request)
        last_message = self.select_last_message(initial_request)
        need_to_rewrite = self.detect_need_to_rewrite(
            state.request, last_message
        )
        rewritten_last_message = self.rewrite_last_message_if_needed(
            initial_request, need_to_rewrite, last_message
        )
        prompt = self.replace_last_message(
            rewritten_last_message,
            initial_request,
            last_message,
            need_to_rewrite,
        )
        state = self.prepare_request_step_3(
            initial_request,
            state,
            need_to_rewrite,
            prompt,
        )

        return super().process(state)

    def detect_need_to_rewrite(
        self, initial_request: RequestState, last_message: str
    ) -> bool:
        new_request = replace(
            initial_request,
            prompt=self.write_detection_prompt(
                last_message, initial_request.model
            ),
        )
        new_request = replace(
            new_request,
            max_tokens=200,
        )
        new_request = replace(
            new_request,
            stop_sequences=["END"],
        )
        print("========= START SG implementation step 1 =============")
        print("Going to detect need to rewrite prompt")
        print("prompt", new_request.prompt)
        result_step_1: RequestResult = self.service.make_request(
            self.execution_spec.auth, new_request
        )
        print("completions:", [seq.text for seq in result_step_1.completions])
        file = pick_right_log_file(new_request.model)
        log_api_request(
            file,
            new_request,
            result_step_1,
            raw_request={},
            prefix="Detection step (3-steps SG)",
        )
        print("========= END SG implementation step 1 =============")
        return self.extract_detection_need_to_rewrite(result_step_1)

    def write_detection_prompt(self, last_message, model) -> str:
        eval_instance_block = MULTI_STEP_PROMPT_STEP_1.format(
            scenario=last_message
        )
        return self.adapt_prompt_to_right_format(eval_instance_block, model)

    def select_last_message(self, initial_request) -> str:
        if "anthropic" in initial_request.model:
            all_messages_appended = initial_request.prompt
            all_messages = self.decompose_anthropic_prompt(
                all_messages_appended
            )
            last_human_message = self.get_last_human_message_anthropic(
                all_messages
            )
            return last_human_message
        else:
            return initial_request.prompt

    def decompose_anthropic_prompt(self, all_messages_appended):
        blocks = all_messages_appended.split(HUMAN_PROMPT)
        all_messages = []
        for one_block in blocks:
            messages_in_block = one_block.split(AI_PROMPT)
            for i, one_message in enumerate(messages_in_block):
                if i == 0:
                    all_messages.append((HUMAN_PROMPT, one_message))
                else:
                    all_messages.append((AI_PROMPT, one_message))
        return all_messages

    def get_last_human_message_anthropic(self, messages: List[Tuple[str, str]]):
        for message in reversed(messages):
            if message[0] == HUMAN_PROMPT:
                return message[1]
        return None

    def adapt_prompt_to_right_format(self, eval_instance_block, model_name):
        prompt = eval_instance_block
        if "anthropic" in model_name:
            prompt = HUMAN_PROMPT + " " + prompt + AI_PROMPT + ""
        return prompt

    def extract_detection_need_to_rewrite(
        self,
        result_step_1: RequestResult,
    ) -> bool:
        assert len(result_step_1.completions) == 1
        detections = sg_demo.is_surrogate_threat_detected(
            detection_completions=[
                seq.text for seq in result_step_1.completions
            ],
            v_goal=THREE_STEPS_SG_IMPLEMENTATION_VERSION_TO_USE,
        )
        assert len(detections) == 1
        return detections[0]

    def rewrite_last_message_if_needed(
        self,
        initial_request,
        need_to_rewrite_prompt,
        last_message,
    ) -> RequestState:
        if not need_to_rewrite_prompt:
            return None

        new_request = replace(
            initial_request,
            prompt=self.write_replacement_prompt(
                last_message, initial_request.model
            ),
        )

        n_tokens = count_tokens(last_message, initial_request.model)
        new_request = replace(new_request, max_tokens=2 * n_tokens)
        stop_sequences = ["END"]
        assert all(
            stop_seq not in initial_request.prompt
            for stop_seq in stop_sequences
        )
        new_request = replace(
            new_request,
            stop_sequences=stop_sequences,
        )
        print("========= START SG implementation step 2 =============")
        print("Going to rewrite prompt")
        print("prompt", new_request.prompt)
        result_step_2: RequestResult = self.service.make_request(
            self.execution_spec.auth, new_request
        )
        print("completions:", [seq.text for seq in result_step_2.completions])
        file = pick_right_log_file(new_request.model)
        log_api_request(
            file,
            new_request,
            result_step_2,
            raw_request={},
            prefix="Translation step (3-steps SG)",
        )
        print("========= END SG implementation step 2 =============")
        rewritten_last_message = self.extract_rewritten_prompt(result_step_2)
        return rewritten_last_message

    def prepare_request_step_3(
        self,
        initial_request,
        state,
        need_to_rewrite_prompt,
        prompt,
    ):
        if need_to_rewrite_prompt:
            rewritten_request = replace(
                initial_request,
                prompt=prompt,
            )
            state = replace(state, request=rewritten_request)
        else:
            state = replace(state, request=initial_request)
        return state

    def write_replacement_prompt(self, last_message, model) -> str:
        eval_instance_block = MULTI_STEP_PROMPT_STEP_2.format(
            scenario=last_message
        )
        return self.adapt_prompt_to_right_format(
            eval_instance_block, model_name=model
        )

    def extract_rewritten_prompt(self, result_step_2: RequestResult) -> str:
        assert len(result_step_2.completions) == 1
        completion = result_step_2.completions[0].text
        if completion.startswith("```"):
            return completion[len("```") :]
        if completion.endswith("```"):
            return completion[len("```") :]
        return completion

    def replace_last_message(
        self,
        rewritten_last_message,
        initial_request,
        last_message,
        need_to_rewrite,
    ) -> str:
        if not need_to_rewrite:
            return None

        if "anthropic" in initial_request.model:
            initial_prompt = initial_request.prompt
            assert (
                len(initial_prompt.split(last_message)) == 2
            ), f"last_message: {last_message}, initial_prompt: {initial_prompt}"
            prompt = initial_prompt.replace(
                last_message, rewritten_last_message
            )
        else:
            prompt = rewritten_last_message
        return prompt


def clean_code_completion(completion: str):
    lines = completion.split("\n")

    # remove lines before the first '```python' statement
    for i, line in enumerate(lines):
        if line.strip() == "```python":
            lines = lines[i + 1 :]
            break

    lines = [line for line in lines if not line.strip().startswith("#")]
    lines = [line for line in lines if not line.strip().startswith("import ")]
    lines = [line for line in lines if not line.startswith("def ")]
    lines = [line for line in lines if not line.startswith("class ")]
    lines = [line for line in lines if not line.strip().startswith("from ")]
    lines = [line for line in lines if not line.strip().startswith("```")]

    lines = [
        line
        for i, line in enumerate(lines)
        if line.startswith("\t")
        or line.startswith("    ")
        or (
            i == 0
            # keep first line if it is not a regular sentence (part of a comment)
            and not line.endswith(".")
            and not line.endswith("!")
            and not line.endswith("?")
            and not (
                line.endswith(":")
                and "if " not in line
                and "for " not in line
                and "while " not in line
                and "except " not in line
                and "else" not in line
                and "elif " not in line
                and "try" not in line
                and "match " not in line
            )
        )
    ]

    # remove comment surrounded by """ """ if at the start of the completion
    content = "\n".join(lines).strip()
    if content.startswith('"""'):
        end = content.find('"""', 3)
        if end != -1:
            lines = content[end + 3 :].split("\n")

    if len(lines) > 0 and not lines[0].startswith("    "):
        lines[0] = "    " + lines[0]

    # remove lines after the last 'return XXX' statement if the next line is empty
    for i, line in enumerate(reversed(lines)):
        if line.strip().startswith("return "):
            if i > 0 and len(lines[-i].strip()) == 0:
                # if "return " not found in the last line and if the previous line was empty
                # then remove all the lines after the last "return", they are a comment
                lines = lines[:-i]
            break

    return "\n".join(lines)


def count_tokens(text: str, model):
    if model.startswith("ft:"):
        model_simplified = model.split(":")[1]
        encoding = tiktoken.encoding_for_model(model_simplified)
    else:
        encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)
