import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from saner_exp.agent import (
    ChatCompletionTokenCounter,
    DashScopeTokenCounter,
    EndpointConfig,
    ModelTransportError,
    ModelProtocolError,
    OpenAICompatibleClient,
    OffsetTokenCounter,
    VLLMTokenCounter,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_openai_client_sends_one_action_policy_and_parses_tool_call():
    captured = {}

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({
            "id": "response-1", "created": 123, "system_fingerprint": "fp",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-1", "type": "function",
                    "function": {"name": "search_text", "arguments": "{\"pattern\":\"needle\"}"},
                }]},
            }],
            "usage": {
                "prompt_tokens": 20, "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 12},
            },
        })

    config = EndpointConfig("http://127.0.0.1:8000/v1", "Qwen3-14B", "secret", 12)
    turn = OpenAICompatibleClient(config, transport).complete(
        [{"role": "user", "content": "find it"}], [{"type": "function"}], max_tokens=2048,
    )
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["body"]["parallel_tool_calls"] is False
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["top_p"] == 1.0
    assert captured["body"]["n"] == 1
    assert captured["body"]["stream"] is False
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert turn.tool_calls[0].name == "search_text"
    assert turn.prompt_tokens == 20 and turn.completion_tokens == 8
    assert turn.cached_prompt_tokens == 12
    assert "secret" not in repr(config)


def test_openai_client_rejects_malformed_response():
    client = OpenAICompatibleClient(
        EndpointConfig("https://example.invalid/v1", "model", ""),
        lambda request, timeout: FakeResponse({"choices": []}),
    )
    with pytest.raises(ModelProtocolError):
        client.complete([], [], max_tokens=1)


def test_openai_client_sends_explicit_thinking_mode():
    captured = {}

    def transport(request, timeout):
        captured["body"] = json.loads(request.data)
        return FakeResponse({
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    config = EndpointConfig(
        "https://example.test/compatible-mode/v1", "qwen-snapshot", "",
        enable_thinking=False,
    )
    OpenAICompatibleClient(config, transport).complete([], [], max_tokens=1)
    assert captured["body"]["enable_thinking"] is False


def test_chat_completion_counter_uses_one_output_token_and_reported_prompt_usage():
    captured = {}

    def transport(request, timeout):
        captured["body"] = json.loads(request.data)
        return FakeResponse({
            "choices": [{
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "x"},
            }],
            "usage": {"prompt_tokens": 42, "completion_tokens": 1},
        })

    counter = ChatCompletionTokenCounter(
        EndpointConfig("https://example.test/v1", "model", ""), transport,
    )
    assert counter([{"role": "user", "content": "hello"}], []) == 42
    assert captured["body"]["max_tokens"] == 1


def test_vllm_token_counter_uses_chat_template_endpoint():
    captured = {}

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return FakeResponse({"count": 220, "max_model_len": 32768, "tokens": []})

    config = EndpointConfig("http://127.0.0.1:8000/v1", "Qwen3-14B", "")
    counter = VLLMTokenCounter(config, transport)
    assert counter([{"role": "user", "content": "hello"}], [{"type": "function"}]) == 220
    assert captured["url"] == "http://127.0.0.1:8000/tokenize"
    assert captured["body"]["add_generation_prompt"] is True
    assert captured["body"]["model"] == "Qwen3-14B"


def test_offset_token_counter_adds_conservative_margin():
    counter = OffsetTokenCounter(lambda messages, tools: 123, 512)
    assert counter([{"role": "user", "content": "hello"}], []) == 635


def test_dashscope_token_counter_includes_messages_tools_and_thinking_mode():
    captured = {}

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        return FakeResponse({"usage": {"input_tokens": 323}})

    config = EndpointConfig(
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "qwen3-max-2026-01-23",
        "secret",
        enable_thinking=False,
    )
    counter = DashScopeTokenCounter(config, transport)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function"}]
    assert counter(messages, tools) == 323
    assert captured["url"] == "https://workspace.cn-beijing.maas.aliyuncs.com/api/v1/tokenizer"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"] == {
        "model": "qwen3-max-2026-01-23",
        "input": {"messages": messages},
        "parameters": {"tools": tools, "enable_thinking": False},
    }


def test_dashscope_message_only_probe_omits_parameters():
    captured = {}

    def transport(request, timeout):
        captured["body"] = json.loads(request.data)
        return FakeResponse({"usage": {"input_tokens": 7}})

    counter = DashScopeTokenCounter(
        EndpointConfig(
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            "qwen3-max-2026-01-23", "", enable_thinking=False,
        ),
        transport,
    )
    assert counter.count_messages([{"role": "user", "content": "hello"}]) == 7
    assert "parameters" not in captured["body"]


def test_dashscope_http_error_reports_provider_reason_and_redacts_key():
    def transport(request, timeout):
        raise HTTPError(
            request.full_url, 400, "Bad Request", {},
            BytesIO(json.dumps({
                "code": "InvalidParameter",
                "message": "unsupported model with sk-secret-value",
            }).encode()),
        )

    counter = DashScopeTokenCounter(
        EndpointConfig(
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            "qwen3-max-2026-01-23", "sk-secret-value",
        ),
        transport,
    )
    with pytest.raises(ModelTransportError) as caught:
        counter.count_messages([])
    assert "InvalidParameter" in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "sk-secret-value" not in str(caught.value)


@pytest.mark.parametrize("url", ["localhost:8000/v1", "https://user:pass@example.test/v1", "file:///x"])
def test_endpoint_rejects_unsafe_or_ambiguous_urls(url):
    with pytest.raises(ValueError):
        EndpointConfig(url, "model", "")
