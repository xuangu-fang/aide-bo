"""Backend for DeepSeek API."""

import json
import logging
import time
import aiohttp
from typing import Optional

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values

logger = logging.getLogger("aide")

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
_session: Optional[aiohttp.ClientSession] = None

DEEPSEEK_TIMEOUT_EXCEPTIONS = (
    aiohttp.ClientError,
    aiohttp.ServerTimeoutError,
    aiohttp.ServerConnectionError,
)

@once
def _setup_deepseek_session(api_key: str):
    global _session
    if _session is None:
        _session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

def query(
    system_message: str | None,
    user_message: str | None,
    api_key: str,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Query the DeepSeek API.
    
    Args:
        system_message: Optional system message
        user_message: Optional user message
        api_key: DeepSeek API key
        func_spec: Optional function specification
        convert_system_to_user: Whether to convert system message to user message
        **model_kwargs: Additional model parameters
    
    Returns:
        Tuple of (output, request_time, input_tokens, output_tokens, info)
    """
    _setup_deepseek_session(api_key)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)

    if func_spec is not None:
        filtered_kwargs["functions"] = [func_spec.as_dict()]
        filtered_kwargs["function_call"] = {"name": func_spec.name}

    data = {
        "messages": messages,
        **filtered_kwargs,
    }

    t0 = time.time()
    response = backoff_create(
        lambda: _session.post(  # type: ignore
            f"{DEEPSEEK_API_BASE}/chat/completions",
            json=data
        ),
        DEEPSEEK_TIMEOUT_EXCEPTIONS,
    )
    req_time = time.time() - t0

    completion = response.json()
    choice = completion["choices"][0]

    if func_spec is None:
        output = choice["message"]["content"]
    else:
        function_call = choice["message"].get("function_call")
        assert function_call, f"function_call is empty, it is not a function call: {choice['message']}"
        assert function_call["name"] == func_spec.name, "Function name mismatch"
        try:
            output = json.loads(function_call["arguments"])
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding the function arguments: {function_call['arguments']}")
            raise e

    usage = completion.get("usage", {})
    in_tokens = usage.get("prompt_tokens", 0)
    out_tokens = usage.get("completion_tokens", 0)

    info = {
        "model": completion.get("model"),
        "created": completion.get("created"),
    }

    return output, req_time, in_tokens, out_tokens, info 