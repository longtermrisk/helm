import copy
import os
from dataclasses import replace

import surrogate_goal_demo
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.executor import Executor
from helm.common.request import RequestResult

import surrogate_goal_demo.analysis.utils.multi_step_SG_implementation as sg_demo


os.environ["TOKENIZERS_PARALLELISM"] = "true"

USE_SINGLE_STEP_SG_IMPLEMENTATION = False
USE_THREE_STEPS_SG_IMPLEMENTATION = True
assert not (
    USE_SINGLE_STEP_SG_IMPLEMENTATION and USE_THREE_STEPS_SG_IMPLEMENTATION
)


def load_single_step_goal_prompt():
    filepath = os.path.join(
        os.path.dirname(surrogate_goal_demo.__file__),
        "../data/messages/measure_p_give_in/without_CoT_with_SG_v106.txt",
    )
    with open(filepath, "r") as f:
        prompt = f.read()
    prefix_split_str = "=== Start of your extra instructions ==="
    suffix_split_str = "=== Start of your regular instructions ==="
    assert suffix_split_str in prompt
    prompt = prompt.split(suffix_split_str)[0].strip() + "\n" + suffix_split_str
    assert prefix_split_str in prompt
    prompt = (
        prefix_split_str + "\n" + prompt.split(prefix_split_str)[-1].strip()
    )
    print(f"Loaded single step goal prompt from {filepath}")
    print(f"Prompt: '{prompt}'")
    return prompt


def load_three_steps_goal_prompts():
    filepath_step_1 = os.path.join(
        os.path.dirname(surrogate_goal_demo.__file__),
        "../data/messages/multi_step_sg_implementation/multi_step_sg_detection_template_v307.txt",
    )
    filepath_step_2 = os.path.join(
        os.path.dirname(surrogate_goal_demo.__file__),
        "../data/messages/multi_step_sg_implementation/multi_step_sg_replacement_v307.txt",
    )
    with open(filepath_step_1, "r") as f:
        prompt_step_1 = f.read()
    with open(filepath_step_2, "r") as f:
        prompt_step_2 = f.read()

    prefix_split_str_step_1 = "USER:"
    assert prefix_split_str_step_1 in prompt_step_1
    prompt_step_1 = prompt_step_1.split(prefix_split_str_step_1)[-1].strip()

    prefix_split_str_step_2 = "USER:"
    assert prefix_split_str_step_2 in prompt_step_2
    prompt_step_2 = prompt_step_2.split(prefix_split_str_step_2)[-1].strip()

    print(
        f"Loaded three steps goal prompt from {filepath_step_1} and"
        f" {filepath_step_2}"
    )
    print(f"Prompt step 1: '{prompt_step_1}'")
    print(f"Prompt step 2: '{prompt_step_2}'")
    return prompt_step_1, prompt_step_2


SINGLE_STEP_PROMPT = (
    load_single_step_goal_prompt()
    if USE_SINGLE_STEP_SG_IMPLEMENTATION
    else None
)
MULTI_STEP_PROMPT_STEP_1, MULTI_STEP_PROMPT_STEP_2 = (
    load_three_steps_goal_prompts()
    if USE_THREE_STEPS_SG_IMPLEMENTATION
    else (None, None)
)


class MultiStepExecutor(Executor):
    def process(self, state: RequestState) -> RequestState:
        self.tokenizer_service = None
        self.adapter = None

        initial_request = copy.deepcopy(state.request)
        need_to_rewrite_prompt = self.detect_need_to_rewrite_prompt(
            state.request
        )
        state = self.rewrite_prompt_if_needed(
            state, initial_request, need_to_rewrite_prompt
        )

        return super().process(state)

    def detect_need_to_rewrite_prompt(
        self, initial_request: RequestState
    ) -> bool:
        new_request = replace(
            initial_request,
            prompt=self.write_detection_prompt(initial_request),
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
        print("========= END SG implementation step 1 =============")
        return self.extract_detection_need_to_rewrite_prompt(result_step_1)

    def write_detection_prompt(self, initial_request) -> str:
        eval_instance_block = MULTI_STEP_PROMPT_STEP_1.format(
            scenario=initial_request.prompt
        )
        return self.adapt_prompt_to_right_format(
            eval_instance_block, initial_request.model
        )

    def adapt_prompt_to_right_format(self, eval_instance_block, model_name):
        prompt = eval_instance_block
        # if isinstance(self.service.client, AnthropicClient):
        if "anthropic" in model_name:
            prompt = "\n\nHuman: " + prompt + "\n\nAssistant:"
        # print(self.service.client)
        return prompt

        # The "clean" method using the following doesn't work because
        # it seems that the adapter is used at two different places and
        # doesn't contain the same information. I get the info from the 2nd use,
        # which contain 'global_prefix=""'.
        # Another thing that this method would need to manage would be
        # the fact that the output_prefix is '\n\nAssistant: The answer is '
        # instead of the '\n\nAssistant:'. So its already a merged of the
        # suffix I want one used for this specific case.

        # if self.tokenizer_service == None:
        #     self.tokenizer_service = TokenizerService(
        #         self.service, self.execution_spec.auth
        #     )
        # if self.adapter == None:
        #     self.adapter: Adapter = AdapterFactory.get_adapter(
        #         CURRENT_RUN_SPEC_ADAPTER_SPEC,
        #         tokenizer_service=self.tokenizer_service,
        #     )
        # # Prompt
        # print("CURRENT_RUN_SPEC_ADAPTER_SPEC", CURRENT_RUN_SPEC_ADAPTER_SPEC)
        # prompt = Prompt(
        #     global_prefix=CURRENT_RUN_SPEC_ADAPTER_SPEC.global_prefix,
        #     instructions_block=[],
        #     train_instance_blocks=[""],
        #     eval_instance_block=eval_instance_block,
        #     instance_prefix="",
        #     substitutions=CURRENT_RUN_SPEC_ADAPTER_SPEC.substitutions,
        # )
        # # Make prompt fit within the context window
        # prompt = self.adapter._make_prompt_fit(prompt)
        # print("prompt", prompt)
        # return prompt.text

    def extract_detection_need_to_rewrite_prompt(
        self,
        result_step_1: RequestResult,
    ) -> bool:
        assert len(result_step_1.completions) == 1
        detections = sg_demo.is_surrogate_threat_detected(
            detection_completions=[
                seq.text for seq in result_step_1.completions
            ],
            v_goal=11,
        )
        assert len(detections) == 1
        return detections[0]

    def rewrite_prompt_if_needed(
        self, state: RequestState, initial_request, need_to_rewrite_prompt
    ) -> RequestState:
        if need_to_rewrite_prompt:
            new_request = replace(
                initial_request,
                prompt=self.write_replacement_prompt(initial_request),
            )
            new_request = replace(
                new_request, max_tokens=2 * len(initial_request.prompt)
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
            print("========= START SG implementation step 2 =============")
            print("Going to rewrite prompt")
            print("prompt", new_request.prompt)
            result_step_2: RequestResult = self.service.make_request(
                self.execution_spec.auth, new_request
            )
            print(
                "completions:", [seq.text for seq in result_step_2.completions]
            )
            print("========= END SG implementation step 2 =============")
            rewritten_request = replace(
                initial_request,
                prompt=self.extract_rewritten_prompt(result_step_2),
            )
            state = replace(state, request=rewritten_request)
        else:
            state = replace(state, request=initial_request)
        return state

    def write_replacement_prompt(self, initial_request: RequestState) -> str:
        eval_instance_block = MULTI_STEP_PROMPT_STEP_2.format(
            scenario=initial_request.prompt
        )
        return self.adapt_prompt_to_right_format(
            eval_instance_block, model_name=initial_request.model
        )

    def extract_rewritten_prompt(self, result_step_2: RequestResult) -> str:
        assert len(result_step_2.completions) == 1
        completion = result_step_2.completions[0].text
        if completion.startswith("```"):
            return completion[len("```") :]
        if completion.endswith("```"):
            return completion[len("```") :]
        return completion
