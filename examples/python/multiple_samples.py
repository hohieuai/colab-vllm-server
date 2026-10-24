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

"""Several samples per prompt, e.g. for pass@k or preference-pair generation.

`n` returns diverse samples in one request, but `n > 1` combined with `seed`
is not reliably reproducible: repeating the same seeded request returned
different samples in 5 of 7 trials (vLLM 0.32). When samples must be
reproducible, send one `n=1` request per seed instead.
"""

from openai import OpenAI

client = OpenAI()
MODEL = "Qwen/Qwen3.8-27B"
messages = [
    {"role": "user", "content": "Suggest a name for a coffee shop. Reply with the name only."}
]
no_thinking = {"chat_template_kwargs": {"enable_thinking": False}}

# Diverse, not reproducible: one request, n samples.
response = client.chat.completions.create(
    model=MODEL, messages=messages, n=4, temperature=1.0, max_tokens=32, extra_body=no_thinking
)
print("n=4:", [choice.message.content for choice in response.choices])

# Diverse and reproducible: one request per seed.
samples = [
    client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1.0,
        seed=seed,
        max_tokens=32,
        extra_body=no_thinking,
    )
    .choices[0]
    .message.content
    for seed in range(4)
]
print("seeds 0-3:", samples)
