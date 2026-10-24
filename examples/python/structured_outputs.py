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

"""Structured outputs: parse the response straight into a Pydantic model."""

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class CalendarEvent(BaseModel):
    name: str
    day: str
    participants: list[str]


completion = client.beta.chat.completions.parse(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {
            "role": "user",
            "content": "Extract the event: Alice and Bob meet for lunch at Noma on Friday.",
        }
    ],
    response_format=CalendarEvent,
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

event = completion.choices[0].message.parsed
print(event)
