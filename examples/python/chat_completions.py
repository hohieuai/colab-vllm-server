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

"""Chat completion with the official OpenAI Python SDK.

The client reads OPENAI_BASE_URL and OPENAI_API_KEY from the environment.
"""

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.7,
    max_tokens=1024,
    # vLLM extension: arguments for the model's chat template.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(response.choices[0].message.content)
print(response.usage)
