"""Serve the frozen experiment reference tokenizer through a local HTTP endpoint."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from transformers import AutoTokenizer


EXPECTED_MODEL = "Qwen3-14B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "tokenizers"
        / "Qwen3-14B-40c069824f42",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._json(200, {"status": "ok", "model": EXPECTED_MODEL})
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/tokenize":
                self._json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("invalid request length")
                body = json.loads(self.rfile.read(length))
                if body.get("model") != EXPECTED_MODEL:
                    raise ValueError("unexpected model")
                messages = body["messages"]
                tools = body.get("tools") or []
                if type(messages) is not list or type(tools) is not list:
                    raise ValueError("messages and tools must be arrays")
                count = len(tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=True,
                    add_generation_prompt=body.get("add_generation_prompt", True),
                    enable_thinking=False,
                ))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)[:300]})
                return
            self._json(200, {"count": count})

        def log_message(self, format, *values):
            return

        def _json(self, status: int, payload: dict):
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({
        "status": "listening", "host": args.host, "port": args.port,
        "model": EXPECTED_MODEL, "tokenizer": str(args.tokenizer.resolve()),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
