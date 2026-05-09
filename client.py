"""Shared OpenAI client utilities for the local MLX server."""

import re
from collections.abc import Generator

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-MLX-4bit"

SYSTEM_PROMPT = (
    "You are a stable smart factory agent. "
    "You identify factory problems, analyze root causes, and present the best "
    "practical solutions for production, quality, safety, maintenance, IoT, "
    "workforce, and manufacturing operations."
)

# Recommended sampling params per Qwen3 official docs.
# top_k is a local-server extension; passed inside extra_body.
_THINKING_TEMPERATURE = 0.6
_THINKING_TOP_P = 0.95
_THINKING_TOP_K = 20
_NO_THINK_TEMPERATURE = 0.7
_NO_THINK_TOP_P = 0.8
_NO_THINK_TOP_K = 20

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_UNUSED_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def make_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key="not-needed")


def resolve_model(client: OpenAI) -> str:
    try:
        models = client.models.list()
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve models from the local MLX server. "
            f"Check that the server is running at {BASE_URL}."
        ) from exc

    model_items = list(models)
    if not model_items:
        raise RuntimeError("The local MLX server returned no models.")

    available_model_ids = [str(item.id) for item in model_items if item.id]
    if not available_model_ids:
        raise RuntimeError("The local MLX server returned models without IDs.")

    if DEFAULT_MODEL in available_model_ids:
        return DEFAULT_MODEL

    return available_model_ids[0]


def stream_reply(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    enable_thinking: bool = False,
) -> Generator[str, None, None]:
    """Yield reply tokens from the MLX server, skipping <think> blocks.

    When *enable_thinking* is False (default) the model is instructed to skip
    chain-of-thought entirely.  When True, <think>...</think> blocks are
    consumed internally and not yielded; only the final answer tokens stream
    out.
    """
    temperature = _THINKING_TEMPERATURE if enable_thinking else _NO_THINK_TEMPERATURE
    top_p = _THINKING_TOP_P if enable_thinking else _NO_THINK_TOP_P
    top_k = _THINKING_TOP_K if enable_thinking else _NO_THINK_TOP_K

    extra_body: dict[str, object] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "top_k": top_k,
    }

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
    )

    in_thinking = False
    buf = ""

    for chunk in stream:
        delta = chunk.choices[0].delta
        content = delta.content or ""

        if not content:
            continue

        buf += content

        # Process buffer, filtering out <think>...</think> blocks.
        # Tags may span multiple chunks, so we use a streaming state machine.
        while buf:
            if in_thinking:
                end_idx = buf.find(_THINK_CLOSE)
                if end_idx >= 0:
                    in_thinking = False
                    buf = buf[end_idx + len(_THINK_CLOSE) :]
                else:
                    buf = ""  # all buffered content is thinking — discard
            else:
                start_idx = buf.find(_THINK_OPEN)
                if start_idx >= 0:
                    to_yield = buf[:start_idx]
                    if to_yield:
                        yield to_yield
                    in_thinking = True
                    buf = buf[start_idx + len(_THINK_OPEN) :]
                else:
                    yield buf
                    buf = ""
