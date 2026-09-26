"""Minimal OpenAI-compatible chat-completions client with no hidden retries."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Callable, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ModelTransportError(RuntimeError):
    """The service could not complete a request; the experiment runner decides retries."""


class ModelProtocolError(RuntimeError):
    """The service returned a response outside the registered wire contract."""


def _http_failure(service: str, error: HTTPError, *secrets: str) -> ModelTransportError:
    """Return a bounded provider error without exposing request headers or bodies."""
    detail = ""
    try:
        payload = json.loads(error.read())
        if type(payload) is dict:
            code = payload.get("code")
            message = payload.get("message")
            parts = [value for value in (code, message) if type(value) is str and value]
            detail = ": ".join(parts)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        pass
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[REDACTED]")
    suffix = f" ({detail[:500]})" if detail else ""
    return ModelTransportError(f"{service} returned HTTP {error.code}{suffix}")


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: int = 600
    temperature: float = 0.0
    enable_thinking: bool | None = None

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain credentials, a query, or a fragment")
        if not self.model:
            raise ValueError("model must be non-empty")
        if type(self.api_key) is not str:
            raise TypeError("api_key must be a string")
        if not 1 <= self.timeout_seconds <= 3600:
            raise ValueError("timeout_seconds must be from 1 through 3600")
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise TypeError("enable_thinking must be a boolean or None")

    @classmethod
    def from_environment(cls, prefix: str) -> "EndpointConfig":
        prefix = prefix.rstrip("_")
        missing = [name for name in ("BASE_URL", "MODEL") if not os.environ.get(f"{prefix}_{name}")]
        if missing:
            raise ValueError(f"missing environment variables for {prefix}: {missing}")
        thinking_value = os.environ.get(f"{prefix}_ENABLE_THINKING")
        if thinking_value is None:
            enable_thinking = None
        elif thinking_value.lower() in {"0", "false"}:
            enable_thinking = False
        elif thinking_value.lower() in {"1", "true"}:
            enable_thinking = True
        else:
            raise ValueError(f"{prefix}_ENABLE_THINKING must be true or false")
        return cls(
            base_url=os.environ[f"{prefix}_BASE_URL"],
            model=os.environ[f"{prefix}_MODEL"],
            api_key=os.environ.get(f"{prefix}_API_KEY", ""),
            enable_thinking=enable_thinking,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments_json: str

    def as_wire(self) -> dict:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments_json},
        }


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    response_id: str | None
    created: int | None
    system_fingerprint: str | None
    cached_prompt_tokens: int | None = None

    def as_message(self) -> dict:
        message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.as_wire() for call in self.tool_calls]
        return message


class ModelClient(Protocol):
    def complete(self, messages: Sequence[dict], tools: Sequence[dict], *, max_tokens: int) -> AssistantTurn: ...


class OpenAICompatibleClient:
    def __init__(self, config: EndpointConfig, transport: Callable = urlopen):
        self.config = config
        self._transport = transport

    def complete(self, messages: Sequence[dict], tools: Sequence[dict], *, max_tokens: int) -> AssistantTurn:
        if not 1 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be from 1 through 2048")
        body = {
            "model": self.config.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": self.config.temperature,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "max_tokens": max_tokens,
        }
        if self.config.enable_thinking is not None:
            body["enable_thinking"] = self.config.enable_thinking
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        request = Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            with self._transport(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as error:
            raise _http_failure("model service", error, self.config.api_key) from None
        except (URLError, TimeoutError, OSError) as error:
            raise ModelTransportError(f"model service connection failed: {type(error).__name__}") from None
        try:
            payload = json.loads(raw)
            if type(payload["choices"]) is not list or len(payload["choices"]) != 1:
                raise ModelProtocolError("chat-completions response must contain exactly one choice")
            choice = payload["choices"][0]
            message = choice["message"]
        except ModelProtocolError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
            raise ModelProtocolError("model service returned an invalid chat-completions response") from None
        if type(message) is not dict:
            raise ModelProtocolError("assistant message must be an object")
        content = message.get("content")
        if content is not None and type(content) is not str:
            raise ModelProtocolError("assistant content must be a string or null")
        calls = []
        raw_calls = message.get("tool_calls") or []
        if type(raw_calls) is not list:
            raise ModelProtocolError("assistant tool_calls must be an array")
        for item in raw_calls:
            try:
                call_id = item["id"]
                function = item["function"]
                name = function["name"]
                arguments = function["arguments"]
            except (KeyError, TypeError):
                raise ModelProtocolError("malformed tool call in assistant response") from None
            if not all(type(value) is str and value for value in (call_id, name, arguments)):
                raise ModelProtocolError("tool-call id, name, and arguments must be non-empty strings")
            calls.append(ToolCall(call_id, name, arguments))
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_prompt_tokens = (
            prompt_details.get("cached_tokens") if type(prompt_details) is dict else None
        )
        if prompt_tokens is not None and type(prompt_tokens) is not int:
            raise ModelProtocolError("prompt token usage must be an integer")
        if completion_tokens is not None and type(completion_tokens) is not int:
            raise ModelProtocolError("completion token usage must be an integer")
        if cached_prompt_tokens is not None and (
            type(cached_prompt_tokens) is not int or cached_prompt_tokens < 0
            or (prompt_tokens is not None and cached_prompt_tokens > prompt_tokens)
        ):
            raise ModelProtocolError("cached prompt token usage must be a valid integer")
        return AssistantTurn(
            content=content,
            tool_calls=tuple(calls),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_id=payload.get("id") if type(payload.get("id")) is str else None,
            created=payload.get("created") if type(payload.get("created")) is int else None,
            system_fingerprint=(payload.get("system_fingerprint")
                                if type(payload.get("system_fingerprint")) is str else None),
            cached_prompt_tokens=cached_prompt_tokens,
        )


class VLLMTokenCounter:
    """Count the exact rendered chat request through vLLM's tokenizer endpoint."""

    def __init__(self, config: EndpointConfig, transport: Callable = urlopen):
        self.config = config
        self._transport = transport
        parsed = urlsplit(config.base_url.rstrip("/"))
        path = parsed.path[:-3] if parsed.path.endswith("/v1") else parsed.path
        self.endpoint = parsed._replace(path=f"{path.rstrip('/')}/tokenize", query="", fragment="").geturl()

    def __call__(self, messages: Sequence[dict], tools: Sequence[dict]) -> int:
        body = {
            "model": self.config.model,
            "messages": list(messages),
            "tools": list(tools),
            "add_generation_prompt": True,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(self.endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            with self._transport(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read())
        except HTTPError as error:
            raise _http_failure("tokenizer service", error, self.config.api_key) from None
        except (URLError, TimeoutError, OSError) as error:
            raise ModelTransportError(f"tokenizer service connection failed: {type(error).__name__}") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ModelProtocolError("tokenizer service returned invalid JSON") from None
        count = payload.get("count") if type(payload) is dict else None
        if type(count) is not int or count < 0:
            raise ModelProtocolError("tokenizer response has no valid token count")
        return count


class ChatCompletionTokenCounter:
    """Measure the exact prompt through a one-token Chat Completions request."""

    def __init__(self, config: EndpointConfig, transport: Callable = urlopen):
        self.client = OpenAICompatibleClient(config, transport)

    def __call__(self, messages: Sequence[dict], tools: Sequence[dict]) -> int:
        turn = self.client.complete(messages, tools, max_tokens=1)
        if turn.prompt_tokens is None:
            raise ModelProtocolError("metering response omitted prompt token usage")
        return turn.prompt_tokens


class OffsetTokenCounter:
    """Add a fixed conservative margin to another token counter."""

    def __init__(self, counter: Callable[[Sequence[dict], Sequence[dict]], int], offset: int):
        if type(offset) is not int or offset < 0:
            raise ValueError("token offset must be a non-negative integer")
        self.counter = counter
        self.offset = offset

    def __call__(self, messages: Sequence[dict], tools: Sequence[dict]) -> int:
        return self.counter(messages, tools) + self.offset


class DashScopeTokenCounter:
    """Count a complete service-model request with DashScope's tokenizer API."""

    def __init__(self, config: EndpointConfig, transport: Callable = urlopen):
        self.config = config
        self._transport = transport
        parsed = urlsplit(config.base_url.rstrip("/"))
        suffix = "/compatible-mode/v1"
        if not parsed.path.endswith(suffix):
            raise ValueError("DashScope base_url must end in /compatible-mode/v1")
        prefix = parsed.path[: -len(suffix)]
        self.endpoint = parsed._replace(
            path=f"{prefix}/api/v1/tokenizer", query="", fragment="",
        ).geturl()

    def __call__(self, messages: Sequence[dict], tools: Sequence[dict]) -> int:
        parameters = {"tools": list(tools)}
        if self.config.enable_thinking is not None:
            parameters["enable_thinking"] = self.config.enable_thinking
        return self._count(messages, parameters)

    def count_messages(self, messages: Sequence[dict]) -> int:
        """Probe message tokenization without optional chat parameters."""
        return self._count(messages, None)

    def _count(self, messages: Sequence[dict], parameters: dict | None) -> int:
        body = {
            "model": self.config.model,
            "input": {"messages": list(messages)},
        }
        if parameters is not None:
            body["parameters"] = parameters
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._transport(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read())
        except HTTPError as error:
            raise _http_failure("tokenizer service", error, self.config.api_key) from None
        except (URLError, TimeoutError, OSError) as error:
            raise ModelTransportError(f"tokenizer service connection failed: {type(error).__name__}") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ModelProtocolError("tokenizer service returned invalid JSON") from None
        try:
            count = payload["usage"]["input_tokens"]
        except (KeyError, TypeError):
            raise ModelProtocolError("tokenizer response has no input token count") from None
        if type(count) is not int or count < 0:
            raise ModelProtocolError("tokenizer response has no valid input token count")
        return count
