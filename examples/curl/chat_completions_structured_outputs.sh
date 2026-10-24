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

# POST /v1/chat/completions with a json_schema response format. The server
# constrains the output to the schema.
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [
      {"role": "user", "content": "Extract the event: Alice and Bob meet for lunch at Noma on Friday."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "calendar_event",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "day": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["name", "day", "participants"],
          "additionalProperties": false
        }
      }
    },
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
