import copy

import json
from datetime import datetime
from helm.common.request import Request
from helm.common.request import RequestResult

USE_SINGLE_STEP_SG_IMPLEMENTATION = True
USE_THREE_STEPS_SG_IMPLEMENTATION = False
USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT = False
assert (
    sum(
        [
            USE_SINGLE_STEP_SG_IMPLEMENTATION,
            USE_THREE_STEPS_SG_IMPLEMENTATION,
            USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT,
        ]
    )
    <= 1
), (
    "Only one of USE_SINGLE_STEP_SG_IMPLEMENTATION, USE_THREE_STEPS_SG_IMPLEMENTATION, "
    "USE_THREE_STEPS_SG_IMPLEMENTATION_WT_FT can be True"
)

ANTHROPIC_CLIENT_LOG_FILE = "anthropic_client.log"
OPENAI_CLIENT_LOG_FILE = "openai_client.log"
VERTEXAI_CLIENT_TEXT_LOG_FILE = "vertexai_client_text.log"
VERTEXAI_CLIENT_CHAT_LOG_FILE = "vertexai_client_chat.log"
MISTRAL_CLIENT_LOG_FILE = "mistral_client.log"


def log_api_request(file, *, request, response, raw_request, prefix=None):
    response = copy.deepcopy(response)
    request = copy.deepcopy(request)
    raw_request = copy.deepcopy(raw_request)

    with open(file, "a") as f:
        f.write(f"Date: {datetime.now()}\n")
        if prefix is not None:
            f.write(f"{prefix}\n")
        if isinstance(request, Request):
            f.write(f"result:\n{json.dumps(request.__dict__, indent=2)}\n")
        else:
            f.write(f"request:\n{str(request)}\n")
        f.write(f"raw_request:\n{json.dumps(raw_request, indent=2)}\n")
        if isinstance(response, RequestResult):
            for i in range(len(response.completions)):
                response.completions[i] = response.completions[i].__dict__
                response.completions[i].pop("tokens", None)
                response.completions[i].pop("multimodal_content", None)
            f.write(f"result:\n{json.dumps(response.__dict__, indent=2)}\n")
        else:
            f.write(f"result:\n{json.dumps(response, indent=2)}\n")


def pick_right_log_file(model):
    if "anthropic" in model:
        return ANTHROPIC_CLIENT_LOG_FILE
    elif "openai" in model:
        return OPENAI_CLIENT_LOG_FILE
    elif "google" in model:
        return VERTEXAI_CLIENT_TEXT_LOG_FILE
    else:
        raise NotImplementedError(f"Unknown model: {model}")
