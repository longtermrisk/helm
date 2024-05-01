import copy
import os
from dataclasses import replace

import tiktoken
from anthropic import HUMAN_PROMPT, AI_PROMPT
from typing import List, Tuple

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.executor import Executor
from helm.benchmark.model_metadata_registry import get_model_metadata
from helm.common.request import RequestResult, Request

import surrogate_goal_demo.analysis.utils.multi_step_SG_implementation as sg_demo
from helm.common.clr_constants import (
    USE_SINGLE_STEP_SG_IMPLEMENTATION,
    USE_THREE_STEPS_SG_IMPLEMENTATION,
    log_api_request,
    pick_right_log_file,
    USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT,
)
from surrogate_goal_demo.shared import constants
from surrogate_goal_demo.shared.external_loading_prompts import (
    load_single_step_sg_implementation_prompt,
    load_three_steps_sg_implementation_prompts,
    THREE_STEPS_SG_IMPLEMENTATION_VERSION_TO_USE,
    TRANSLATION_MODEL,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


SINGLE_STEP_PROMPT = (
    load_single_step_sg_implementation_prompt()
    if USE_SINGLE_STEP_SG_IMPLEMENTATION
    else None
)
MULTI_STEP_PROMPT_STEP_1, MULTI_STEP_PROMPT_STEP_2 = (
    load_three_steps_sg_implementation_prompts(
        sg_version=None,
        vanilla=USE_THREE_STEPS_SG_IMPLEMENTATION,
        wt_ft=USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT,
    )
    if (
        USE_THREE_STEPS_SG_IMPLEMENTATION
        or USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT
    )
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
        self, initial_request: Request, last_message: str
    ) -> bool:
        assert (
            "messages" not in initial_request.__dict__
            or initial_request.__dict__["messages"] is None
        ), (
            f"==> messages in initial_request: \n'{initial_request.messages}'\n\n"
            f"==> while prompt: \n'{initial_request.prompt}'"
        )

        new_request = replace(
            copy.deepcopy(initial_request),
            prompt=self.write_detection_prompt(
                last_message, initial_request.model
            ),
        )
        new_request = replace(
            new_request,
            max_tokens=200,
        )
        stop_sequences = ["END"]
        assert all(
            stop_seq not in initial_request.prompt
            for stop_seq in stop_sequences
        )
        new_request = replace(
            new_request,
            stop_sequences=stop_sequences,
        )
        new_request = replace(
            new_request,
            temperature=1.0,
        )
        print("========= START SG implementation step 1 =============")
        print("Going to detect need to rewrite prompt")
        print("prompt", new_request.prompt)
        surrogate_threat_detected = None
        result_step_1 = None
        for trial_i in range(constants.MAX_COMPLETION_TRIALS):
            if trial_i > 0:
                print(f"Trial {trial_i}, retrying completion")
                print("Previous completion invalid: ", result_step_1)
            new_request = replace(
                new_request,
                caching_index=trial_i,
            )
            result_step_1: RequestResult = self.service.make_request(
                self.execution_spec.auth, new_request
            )
            print(
                "completions:", [seq.text for seq in result_step_1.completions]
            )
            file = pick_right_log_file(new_request.model)
            log_api_request(
                file,
                request=new_request,
                response=result_step_1,
                raw_request={},
                prefix="Detection step (3-steps SG)",
            )
            surrogate_threat_detected = self.extract_detection_need_to_rewrite(
                result_step_1
            )
            if surrogate_threat_detected is not None:
                break

        if surrogate_threat_detected is None:
            return False
            # raise ValueError(
            #     "No completion was obtained for the detection prompt"
            # )
        print("========= END SG implementation step 1 =============")
        return surrogate_threat_detected

    def write_detection_prompt(self, last_message, model) -> str:
        eval_instance_block = MULTI_STEP_PROMPT_STEP_1.format(
            scenario=last_message
        )
        return self.adapt_prompt_to_right_format(eval_instance_block, model)

    def select_last_message(self, initial_request) -> str:
        if (
            "anthropic" in initial_request.model
            and "claude-3" not in initial_request.model
        ):
            raise DeprecationWarning()
            # all_messages_appended = initial_request.prompt
            # all_messages = self.decompose_anthropic_prompt(
            #     all_messages_appended
            # )
            # last_human_message = self.get_last_human_message_anthropic(
            #     all_messages
            # )
            # return last_human_message
        else:
            return initial_request.prompt

    # def decompose_anthropic_prompt(self, all_messages_appended):
    #     blocks = all_messages_appended.split(HUMAN_PROMPT)
    #     all_messages = []
    #     for one_block in blocks:
    #         messages_in_block = one_block.split(AI_PROMPT)
    #         for i, one_message in enumerate(messages_in_block):
    #             if i == 0:
    #                 all_messages.append((HUMAN_PROMPT, one_message))
    #             else:
    #                 all_messages.append((AI_PROMPT, one_message))
    #     return all_messages
    #
    # def get_last_human_message_anthropic(self, messages: List[Tuple[str, str]]):
    #     for message in reversed(messages):
    #         if message[0] == HUMAN_PROMPT:
    #             return message[1]
    #     return None

    def adapt_prompt_to_right_format(self, eval_instance_block, model_name):
        prompt = eval_instance_block
        if "anthropic" in model_name and "claude-3" not in model_name:
            prompt = HUMAN_PROMPT + " " + prompt + AI_PROMPT + ""
        return prompt

    def extract_detection_need_to_rewrite(
        self,
        result_step_1: RequestResult,
    ) -> bool:
        if len(result_step_1.completions) == 0:
            return False

        detection_completions = [seq.text for seq in result_step_1.completions]
        detections = sg_demo.is_surrogate_threat_detected(
            detection_completions=detection_completions,
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
        if not need_to_rewrite_prompt or need_to_rewrite_prompt is None:
            return None

        new_request = replace(
            copy.deepcopy(initial_request),
            prompt=self.write_replacement_prompt(
                last_message, initial_request.model
            ),
        )

        n_tokens = count_tokens(last_message, initial_request.model)
        n_tokens_to_request = 2 * n_tokens
        if "gpt-3.5" in initial_request.model:
            n_tokens_to_request = min(n_tokens_to_request, 4000)
        new_request = replace(new_request, max_tokens=n_tokens_to_request)
        stop_sequences = ["END"]
        assert all(
            stop_seq not in initial_request.prompt
            for stop_seq in stop_sequences
        )
        new_request = replace(
            new_request,
            stop_sequences=stop_sequences,
        )
        new_request = replace(
            new_request,
            temperature=1.0,
        )
        print("========= START SG implementation step 2 =============")
        print("Going to rewrite prompt")
        print("prompt", new_request.prompt)
        result_step_2 = None
        success = False
        for trial_i in range(constants.MAX_COMPLETION_TRIALS):
            if trial_i > 0:
                print(f"Trial {trial_i}, retrying completion")
                print("Previous translation invalid: ", result_step_2)
            new_request = replace(
                new_request,
                caching_index=trial_i,
            )
            if USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT:
                new_request = replace(
                    new_request,
                    model=TRANSLATION_MODEL,
                )
            result_step_2: RequestResult = self.service.make_request(
                self.execution_spec.auth, new_request
            )
            print(
                "completions:", [seq.text for seq in result_step_2.completions]
            )
            file = pick_right_log_file(new_request.model)
            log_api_request(
                file,
                request=new_request,
                response=result_step_2,
                raw_request={},
                prefix="Translation step (3-steps SG)",
            )
            assert len(result_step_2.completions) == 1
            completion = result_step_2.completions[0].text
            (
                success,
                raw_scenario_text_wt_vanilla_threat,
            ) = sg_demo.extract_translation_from_completion(
                completion,
                initial_request.prompt,
            )
            if success:
                break

        # if not success:
        #     raise ValueError(
        #         "No valid completion was obtained for the translation prompt"
        #     )

        print("========= END SG implementation step 2 =============")
        if success:
            return raw_scenario_text_wt_vanilla_threat
        else:
            return new_request.prompt

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

    def replace_last_message(
        self,
        rewritten_last_message,
        initial_request,
        last_message,
        need_to_rewrite,
    ) -> str:
        if not need_to_rewrite:
            return None

        if (
            "anthropic" in initial_request.model
            and "claude-3" not in initial_request.model
        ):
            raise DeprecationWarning()
            # initial_prompt = initial_request.prompt
            # assert (
            #     len(initial_prompt.split(last_message)) == 2
            # ), f"last_message: {last_message}, initial_prompt: {initial_prompt}"
            # prompt = initial_prompt.replace(
            #     last_message, rewritten_last_message
            # )
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
    # if model.startswith("ft:"):
    #     model = model.split(":")[1]
    #
    # encoding_name = get_model_metadata(model).tokenizer_name
    # encoding = tiktoken.get_encoding(encoding_name)
    # tokens = encoding.encode(text)
    # return len(tokens)

    words = (
        text.replace("\n", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(".", " ")
        .replace("_", " ")
        .replace("        ", " ")
        .replace("       ", " ")
        .replace("      ", " ")
        .replace("     ", " ")
        .replace("    ", " ")
        .replace("   ", " ")
        .replace("  ", " ")
        .split(" ")
    )
    n_tokens = int(len(words) / 0.75)
    return n_tokens
