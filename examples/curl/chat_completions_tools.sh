#!/usr/bin/env bash
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

# POST /v1/chat/completions with tools. The call is in choices[0].message.tool_calls.
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [
      {"role": "user", "content": "Say hello to the world."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "say_hello",
          "description": "Greet someone by name.",
          "parameters": {
            "type": "object",
            "properties": {
              "name": {"type": "string", "description": "Who to greet"}
            },
            "required": ["name"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 4096
  }'
