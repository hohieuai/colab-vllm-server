# Call functions

This guide shows how to let the model call functions that you define. The model returns the
function name and arguments in `tool_calls`. Your code runs the function and sends the result
back, and the model uses it in its answer.

## Run a complete tool call

```python
import json

from openai import OpenAI

client = OpenAI()

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
response = client.chat.completions.create(model="Qwen/Qwen3.8-27B", messages=messages, tools=tools)
message = response.choices[0].message
messages.append(message)

for call in message.tool_calls or []:
    result = get_weather(**json.loads(call.function.arguments))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

final = client.chat.completions.create(model="Qwen/Qwen3.8-27B", messages=messages, tools=tools)
print(final.choices[0].message.content)
```

The runnable versions are [`examples/python/tool_calling.py`](../../examples/python/tool_calling.py)
and [`examples/curl/chat_completions_tools.sh`](../../examples/curl/chat_completions_tools.sh).

## Steps

1. Describe each function with a `name`, a `description`, and its `parameters` as a JSON Schema.
   The model decides when to call a function from its description, so write it as you would for
   a colleague.
2. Send the request with `tools`. If the model calls a function, `finish_reason` is `tool_calls`.
3. Parse `function.arguments` with `json.loads`. The field is a JSON string, not an object.
4. Append the assistant message. Then append one `tool` message per call, with the matching
   `tool_call_id`.
5. Send the conversation again. The model writes its answer from the results.

With thinking on, the model may reason before the call. The reasoning arrives in
`message.reasoning`, and `content` is empty.

## Force or prevent a call

To control whether the model calls a function, set `tool_choice`. Use `"required"` to force a
call, `"none"` to prevent one, or name a function to force that function. The
[API reference](../reference/api.md#tool-choice) lists the forms.

## Retry a malformed call

The server parses calls with vLLM's `qwen3_coder` parser. If the model writes a malformed call,
the server returns it as plain `content`, with no `tool_calls`. In a pipeline that depends on
tool use, check that `tool_calls` is present, and send the request again if it is missing.

## Related pages

- [Get structured outputs](get-structured-outputs.md)
- [Validation results](../reference/validation-results.md)
