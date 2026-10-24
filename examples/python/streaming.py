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

"""Streaming chat completion."""

from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Write a four-line poem about rain."}],
    max_tokens=512,
    stream=True,
    stream_options={"include_usage": True},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.usage:
        print(f"\n\n{chunk.usage.completion_tokens} completion tokens")
