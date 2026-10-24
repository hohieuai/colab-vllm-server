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

"""Thinking mode: the reasoning comes back separately from the answer."""

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {
            "role": "user",
            "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost?",
        }
    ],
    max_tokens=8192,
    # reasoning_effort: "low" | "medium" | "xhigh" (template default: xhigh)
    extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}},
)

choice = response.choices[0]
# `reasoning` is a vLLM field, not part of the OpenAI schema; the SDK keeps it
# as an extra attribute on the message.
print("Reasoning:", getattr(choice.message, "reasoning", None))
print("Answer:", choice.message.content)
if choice.finish_reason == "length":
    print("Warning: truncated by max_tokens")
