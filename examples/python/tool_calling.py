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

"""Function calling: the model requests a tool, we run it and send the result back."""

import json

from openai import OpenAI

client = OpenAI()
MODEL = "Qwen/Qwen3.8-27B"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str) -> dict:
    return {"city": city, "temperature_c": 31, "condition": "thunderstorms"}


messages = [{"role": "user", "content": "What is the weather like in Hanoi today?"}]
response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools, max_tokens=4096
)
message = response.choices[0].message
messages.append(message)

for call in message.tool_calls or []:
    result = get_weather(**json.loads(call.function.arguments))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=4096)
print(final.choices[0].message.content)
