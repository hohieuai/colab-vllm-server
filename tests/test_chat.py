# Copyright 2026 The colab-vllm-server Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`serve chat` against a local server that speaks the OpenAI streaming API."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from colab_vllm_server import cli

MODEL = "Qwen/Qwen3.8-27B"


class OpenAIStub(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []

    def log_message(self, *args):
        pass

    def _authorized(self):
        if self.headers.get("Authorization") != "Bearer vllm":
            self.send_response(401)
            self.end_headers()
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        body = json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_POST(self):
        if not self._authorized():
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        OpenAIStub.requests.append(request)
        thinking = request["chat_template_kwargs"]["enable_thinking"]
        # After thinking, the model opens its answer with blank lines.
        opening = "\n\nHello, " if thinking else "Hello, "
        deltas = ([{"reasoning": "The user greets me."}] if thinking else []) + [
            {"content": opening},
            {"content": "world!"},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i, delta in enumerate(deltas):
            last = i == len(deltas) - 1
            chunk = {
                "object": "chat.completion.chunk",
                "model": MODEL,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": "stop" if last else None}
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage = {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
        chunk = {"object": "chat.completion.chunk", "model": MODEL, "choices": [], "usage": usage}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    cli.save_state(
        "demo",
        {"port": server.server_address[1], "api_key": "vllm", "served_model_name": MODEL},
    )
    OpenAIStub.requests = []
    yield
    server.shutdown()


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["serve", "chat", "demo", *argv])
    cli.main()


def test_chat_prints_the_streamed_answer(endpoint, monkeypatch, capsys):
    run(monkeypatch, "Hello!")
    out, err = capsys.readouterr()
    assert out == "Hello, world!\n"
    assert "3 tokens" in err and "finish_reason=stop" in err
    request = OpenAIStub.requests[0]
    assert request["model"] == MODEL
    assert request["messages"] == [{"role": "user", "content": "Hello!"}]
    assert request["max_tokens"] == 1024
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_with_think_shows_reasoning_on_stderr(endpoint, monkeypatch, capsys):
    run(monkeypatch, "--think")
    out, err = capsys.readouterr()
    assert out == "Hello, world!\n"
    assert "[thinking] The user greets me." in err
    request = OpenAIStub.requests[0]
    assert request["max_tokens"] == 8192
    assert request["messages"][0]["content"].startswith("Hello!")


def test_chat_exits_when_the_endpoint_is_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="not running"):
        run(monkeypatch)
